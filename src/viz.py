import cv2
import numpy as np

PALETTE_WIDTH = 32 * 5
PALETTE_HEIGHT = 32
COLUMN_PADDING_WIDTH = 10
ROW_PADDING_HEIGHT = 5
TITLE_BAR_HEIGHT = 20

def colors_to_img(colors):
    imgs = np.transpose(colors, (0, 2, 1))            # (N, C, W) -> (N, W, C)
    imgs = np.expand_dims(imgs, axis=1)               # Add height dimension - (N, H, W, C)
    imgs = np.repeat(imgs, PALETTE_WIDTH / 5, axis=2) # Repeat colors horiziontaly to get image of PALETTE_WIDTH width.
    imgs = np.tile(imgs, (1, PALETTE_HEIGHT, 1, 1))   # Repeat colors vertically to get image of PALETTE_HEIGHT height.
    row_padding = np.zeros((imgs.shape[0], ROW_PADDING_HEIGHT, ) + imgs.shape[2:])
    imgs = np.concatenate((imgs, row_padding), axis=1)    # Add padding between each palette so they're sepearated.
    imgs = np.reshape(imgs, (-1, PALETTE_WIDTH, 3))   # Merge N and H dimensions, so color palettes are now in a single image.
    imgs = imgs[:-ROW_PADDING_HEIGHT, :, :]           # Remove padding after last color palette.
    imgs = (imgs * 255).astype(np.uint8)              # (0, 1) range to (0, 255) range
    return imgs


def create_comparison_img(generated_colors, generated_colors_avg, real_colors):
    # Add padding between images.
    padding = np.zeros((generated_colors_avg.shape[0], COLUMN_PADDING_WIDTH, 3), dtype=np.uint8)
    comparison_img = np.concatenate([generated_colors, padding, generated_colors_avg, padding, real_colors], axis=1)

    # Add title bar
    title_bar = np.zeros((TITLE_BAR_HEIGHT, comparison_img.shape[1], 3), dtype=np.uint8)
    comparison_img = np.concatenate([title_bar, comparison_img], axis=0)

    # Text properties.
    font = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_COLOR = (255, 255, 255)
    font_scale = 0.5
    font_thickness = 1
    img_width = comparison_img.shape[1]

    # Add title for each column of colors.
    text_size, _ = cv2.getTextSize(text='GEN CURRENT', fontFace=font, fontScale=font_scale, thickness=font_thickness)
    text_position = (int(img_width / 6 - text_size[0] / 2), int(TITLE_BAR_HEIGHT // 2 + text_size[1] / 2))
    cv2.putText(comparison_img, 'GEN CURRENT', text_position, font, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(text='GEN AVG', fontFace=font, fontScale=font_scale, thickness=font_thickness)
    text_position = (int(img_width / 2 - text_size[0] / 2), int(TITLE_BAR_HEIGHT // 2 + text_size[1] / 2))
    cv2.putText(comparison_img, 'GEN AVG', text_position, font, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(text='REAL', fontFace=font, fontScale=font_scale, thickness=font_thickness)
    text_position = (int(img_width / 6 * 5 - text_size[0] / 2), int(TITLE_BAR_HEIGHT // 2 + text_size[1] / 2))
    cv2.putText(comparison_img, 'REAL', text_position, font, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)

    return comparison_img