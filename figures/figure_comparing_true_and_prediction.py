import numpy as np
from PIL import Image, ImageChops
import pandas as pd
import texfig  # import texfig first to configure Matplotlib's backend

SELECTED_INDICES = [500, 700, 1000, 1200]


def main():

    dataframe_original = pd.read_csv("dataset.csv")
    dataframe_unet = pd.read_csv("dataset_unet.csv")

    ratio = 1.25

    fig, ax = texfig.subplots(ratio=ratio, nrows=5, ncols=4)

    for i, index in enumerate(SELECTED_INDICES):
        path = dataframe_original["images_vel_x"][index]
        image_vel_x = Image.open(path)

        path = dataframe_original["images_vel_y"][index]
        image_vel_y = Image.open(path)

        path = dataframe_unet["images_unet"][index]
        image_pressure_unet = Image.open(path).convert("L")

        path = dataframe_original["images_pressure"][index]
        image_pressure_original = Image.open(path).convert("L")

        # buffer1 = np.asarray(image_pressure_original.resize(image_pressure_unet.size))
        # buffer2 = np.asarray(image_pressure_unet)
        # buffer3 = np.ones_like(buffer1) - buffer1 + buffer2
        # difference = Image.fromarray(buffer3)
        difference = ImageChops.difference(
            image_pressure_original.resize(image_pressure_unet.size),
            image_pressure_unet,
        )
        difference_gray = difference.convert("L")

        ax[0, i].imshow(image_vel_x)
        ax[1, i].imshow(image_vel_y)
        ax[2, i].imshow(image_pressure_unet, cmap="gray", vmin=0, vmax=255)
        ax[3, i].imshow(
            image_pressure_original.resize(image_pressure_unet.size),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        ax[4, i].imshow(difference_gray, cmap="gray_r", vmin=0, vmax=255)
        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
        ax[3, i].axis("off")
        ax[4, i].axis("off")

        ax[0, 0].text(
            0.06,
            0.35,
            r"$u$-velocity",
            rotation="vertical",
            transform=ax[0, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[1, 0].text(
            0.06,
            0.35,
            r"$v$-velocity",
            rotation="vertical",
            transform=ax[1, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[2, 0].text(
            0.06,
            0.33,
            r"Predicted $Cp$",
            rotation="vertical",
            transform=ax[2, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[3, 0].text(
            0.06,
            0.25,
            r"True $Cp$",
            rotation="vertical",
            transform=ax[3, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[4, 0].text(
            0.06,
            0.25,
            r"Difference",
            rotation="vertical",
            transform=ax[4, 0].transAxes,
            ha="center",
            size=10,
        )

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.02, wspace=0.02
    )

    texfig.savefig(
        "Fig_Cp_field_prediction", dpi=1000, bbox_inches="tight", pad_inches=0
    )


if __name__ == "__main__":
    main()
