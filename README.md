# BACKBONE-UNET

ImageNet 1-crop error rates (224x224)

================================  =============   =============       ===========
Network                           Top-1 error     Top-5 error           PARAMS
================================  =============   =============       ===========

VGG-11                            30.98           11.37                     9M
VGG-13                            30.07           10.75                     9M
VGG-16                            28.41           9.62                      14M
VGG-19                            27.62           9.12                      20M
VGG-11 with batch normalization   29.62           10.19                     9M
VGG-13 with batch normalization   28.45           9.63                      9M
VGG-16 with batch normalization   26.63           8.50                      14M
VGG-19 with batch normalization   25.76           8.15                      20M

ResNet-18                         30.24           10.92                     11M
ResNet-34                         26.70           8.58                      21M
ResNet-50                         23.85           7.13                      23M
ResNet-101                        22.63           6.44                      42M            
ResNet-152                        21.69           5.94                      58M
ResNeXt-50-32x4d                  22.38           6.30                      22M
ResNeXt-101-32x8d                 20.69           5.47                      42M



Densenet-121                      25.35           7.83                      6M
Densenet-169                      24.00           7.00                      12M
Densenet-201                      22.80           6.43                      18M
Densenet-161                      22.35           6.20                      26M


MobileNet V2                      28.12           9.71                      2M



================================  =============   =============

