from torch import nn, cat, stack, no_grad
from torchvision.models.mobilenetv3 import InvertedResidualConfig,Optional,Callable,Any,InvertedResidual,Conv2dNormActivation
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, _mobilenet_v3_conf
from torchvision.models.squeezenet import Fire
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._api import WeightsEnum, Weights, register_model
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.transforms._presets import ImageClassification
from collections.abc import Sequence
from sklearn.model_selection import KFold
from dataUtils import *
from functools import partial
from torch import Tensor, flatten
from torch.utils.data import DataLoader, Subset
from typing import List, Any, Optional
import torch.nn.init as init

def train(model, loss_fn, optimizer, epochs, train_loader, val_loader, device='cuda'):
    epochs_train_loss = []
    epochs_val_loss = [] 
    for epoch in range(1,epochs+1):
        print(f'EPOCH {epoch}')
        running_train_loss = 0
        running_vall_loss = 0
        # training
        model.train(True)
        for x,area,ate,are in train_loader:
            optimizer.zero_grad()
            y_hat = model(x.to(device),area.to(device))
            y = stack([ate,are],dim=1).to(device).float()
            train_loss = loss_fn(y_hat.float(), y)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        train_loss_value = running_train_loss/len(train_loader)
        # validation
        model.eval() 
        with no_grad(): 
            for xv,areav,atev,arev in val_loader: 
                y_hat = model(xv.to(device),areav.to(device))
                yv = stack([atev,arev],dim=1).to(device).float()
                val_loss = loss_fn(y_hat.float(), yv)
                running_vall_loss += val_loss.item()
            val_loss_value = running_vall_loss/len(val_loader)
        print(f'\ttrain loss:\t {train_loss_value}\n\tval. loss:\t {val_loss_value}')
        epochs_train_loss += [train_loss_value]
        epochs_val_loss += [val_loss_value]
    return {'train': epochs_train_loss, 'validation': epochs_val_loss} 

def test_error(model, testset, device="cpu", loss_fn=nn.MSELoss()):
    testloader = DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    test_loss = 0.0
    model.to(device)
    model.eval()
    with no_grad():
        for data in testloader:
            x, area, ate, are = data
            x = x.to(device)
            area = area.to(device)
            ate = ate.to(device)
            are = are.to(device)
            outputs = model(x,area)
            labels = stack([ate,are],dim=1).to(device).float()
            loss = loss_fn(outputs, labels)
            test_loss += loss.cpu().numpy()
    return test_loss / len(testloader)

def kCV(k, dataset, model, loss_fn, optimizer, epochs, batch_size, device='cuda'):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_loss = 0
    all_loss = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        train_subset = WrapperDataset(Subset(dataset, train_ids),transform=train_transform)
        val_subset = WrapperDataset(Subset(dataset, val_ids))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2)
        epochs_loss = train(model,loss_fn,optimizer,epochs,train_loader,val_loader,device)
        val_loss = epochs_loss['validation'][-1]
        all_loss += [epochs_loss]
        cv_loss += val_loss
    cv_loss /= k
    return {'CV-loss':cv_loss, 'all-loss':all_loss}

class AlexNet(nn.Module):
    def __init__(self, activation_function=nn.ReLU(), dropout=0.5):
        super(AlexNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,96, kernel_size=11, stride=4, padding=1),
            activation_function, nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96,256, kernel_size=5, padding=2),  activation_function,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256,384, kernel_size=3, padding=1), activation_function,
            nn.Conv2d(384,384, kernel_size=3, padding=1), activation_function,
            nn.Conv2d(384,256, kernel_size=3, padding=1), activation_function,
            nn.MaxPool2d(kernel_size=3, stride=3), nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(4096+1,4096), activation_function, nn.Dropout(p=dropout),
            nn.Linear(4096,4096) , activation_function, nn.Dropout(p=dropout),
            nn.Linear(4096,2)
        )

    def forward(self, map, area):
        cnn_out = self.cnn(map)
        combined = cat([cnn_out,area.view(-1,1).float()],dim=1)
        return self.linear(combined)

# MobileNetV3 #
class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: list[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                1,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels+1, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, map: Tensor, area:Tensor) -> Tensor:
        x = self.features(map)

        x = self.avgpool(x)
        cnn_out = flatten(x, 1)

        combined = cat([cnn_out,area.view(-1,1).float()],dim=1)
        x = self.classifier(combined)

        return x

    def forward(self, map: Tensor, area:Tensor) -> Tensor:
        return self._forward_impl(map,area)
    
def mobilenet_v3_small(
    *, weights: Optional[MobileNet_V3_Small_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V3_Small_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V3_Small_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenet.MobileNetV3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V3_Small_Weights
        :members:
    """
    weights = MobileNet_V3_Small_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)

def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

# SqueezeNet1.1 #
class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )

class SqueezeNet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

        self.finalClassifier = nn.Sequential(
            nn.Linear(3,16),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16,64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: Tensor, area: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        combined = cat([flatten(x,1),area.view(-1,1).float()],dim=1)
        return self.finalClassifier(combined)

def _squeezenet(
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SqueezeNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SqueezeNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/49#issuecomment-277560717",
    "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
}

class SqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (17, 17),
            "num_params": 1235496,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 58.178,
                    "acc@5": 80.624,
                }
            },
            "_ops": 0.349,
            "_file_size": 4.729,
        },
    )
    DEFAULT = IMAGENET1K_V1

@handle_legacy_interface(weights=("pretrained", SqueezeNet1_1_Weights.IMAGENET1K_V1))
def squeezenet1_1(
    *, weights: Optional[SqueezeNet1_1_Weights] = None, progress: bool = True, **kwargs: Any
) -> SqueezeNet:
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.

    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_1_Weights
        :members:
    """
    weights = SqueezeNet1_1_Weights.verify(weights)
    return _squeezenet(weights, progress, **kwargs)