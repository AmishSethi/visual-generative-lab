"""
SongUNet implementation specifically for position conditioning.
"""

from vgl.unet_models_song import SongUNet


def SongUNet_Position_XL(**kwargs):
    return SongUNet(
        conditioning_type='position',
        model_channels=192,
        channel_mult=[1,2,3,4],
        num_blocks=3,
        **kwargs
    )

def SongUNet_Position_L(**kwargs):
    return SongUNet(
        conditioning_type='position',
        model_channels=192,
        channel_mult=[1,2,2,2],
        num_blocks=3,
        **kwargs
    )

def SongUNet_Position_B(**kwargs):
    return SongUNet(
        conditioning_type='position',
        model_channels=128,
        channel_mult=[1,2,2,2],
        num_blocks=4,
        **kwargs
    )

def SongUNet_Position_S(**kwargs):
    return SongUNet(
        conditioning_type='position',
        model_channels=128,
        channel_mult=[1,2,2],
        num_blocks=3,
        **kwargs
    )


# Model name mappings
SongUNet_Position_models = {
    'SongUNet-Position-XL': SongUNet_Position_XL,
    'SongUNet-Position-L': SongUNet_Position_L,
    'SongUNet-Position-B': SongUNet_Position_B,
    'SongUNet-Position-S': SongUNet_Position_S,
}