"""
SongUNet implementation specifically for rotation conditioning.
"""

from vgl.unet_models_song import SongUNet


def SongUNet_Rotation_XL(**kwargs):
    return SongUNet(
        conditioning_type='rotation',
        model_channels=192,
        channel_mult=[1,2,3,4],
        num_blocks=3,
        **kwargs
    )

def SongUNet_Rotation_L(**kwargs):
    return SongUNet(
        conditioning_type='rotation',
        model_channels=192,
        channel_mult=[1,2,2,2],
        num_blocks=3,
        **kwargs
    )

def SongUNet_Rotation_B(**kwargs):
    return SongUNet(
        conditioning_type='rotation',
        model_channels=128,
        channel_mult=[1,2,2,2],
        num_blocks=4,
        **kwargs
    )

def SongUNet_Rotation_S(**kwargs):
    return SongUNet(
        conditioning_type='rotation',
        model_channels=128,
        channel_mult=[1,2,2],
        num_blocks=3,
        **kwargs
    )


# Model name mappings
SongUNet_Rotation_models = {
    'SongUNet-Rotation-XL': SongUNet_Rotation_XL,
    'SongUNet-Rotation-L': SongUNet_Rotation_L,
    'SongUNet-Rotation-B': SongUNet_Rotation_B,
    'SongUNet-Rotation-S': SongUNet_Rotation_S,
}