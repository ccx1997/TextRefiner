import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from STN import STN


def test_batch_transform():
    transformer = STN(
        output_img_size=(32, 100),
        num_control_points=6,
        margins=[0.1,0.1]
    )
    test_input_ctrl_pts = np.array([
        [
            [-0.8, -0.2], [0.0, -0.8], [0.8, -0.2],
            [-0.8, 0.8], [0.0, 0.2], [0.8, 0.8]
        ],
        [
            [-0.8, -0.8], [0.0, -0.2], [0.8, -0.8],
            [-0.8, 0.3], [0.0, 0.8], [0.8, 0.3]
        ],
        [
            [-0.8, -0.8], [0.0, -0.8], [0.8, -0.8],
            [-0.8, 0.8], [0.0, 0.8], [0.8, 0.8],
        ]
    ], dtype=np.float32)

    test_im = Image.open('/workspace/xqq/aster/data/test_image.jpg').resize((128, 128))
    test_image_array = np.array(test_im).transpose((2,0,1))
    test_image_array = np.array([test_image_array, test_image_array, test_image_array])
    test_images = torch.Tensor(test_image_array)
    test_images = (test_images / 128.0) - 1.0

    rectified_images,source_coordinate = transformer(test_images,torch.Tensor(test_input_ctrl_pts))
    rectified_images = np.transpose(rectified_images.detach().numpy(),(0,2,3,1))
    outputs = {'sampling_grid':source_coordinate.detach().numpy(),'rectified_images':rectified_images}

    output_ctrl_pts = transformer.target_control_points

    rectified_images_ = (outputs['rectified_images'] + 1.0) * 128.0


    if True:
        plt.figure()
        plt.subplot(3, 4, 1)
        plt.scatter(test_input_ctrl_pts[0, :, 0], test_input_ctrl_pts[0, :, 1])
        plt.subplot(3, 4, 2)
        plt.scatter(output_ctrl_pts[:, 0], output_ctrl_pts[:, 1])
        plt.subplot(3, 4, 3)
        plt.scatter(outputs['sampling_grid'][0, :, 0], outputs['sampling_grid'][0, :, 1], marker='+')
        plt.subplot(3, 4, 4)
        plt.imshow(rectified_images_[0].astype(np.uint8))

        plt.subplot(3, 4, 5)
        plt.scatter(test_input_ctrl_pts[1, :, 0], test_input_ctrl_pts[1, :, 1])
        plt.subplot(3, 4, 6)
        plt.scatter(output_ctrl_pts[:, 0], output_ctrl_pts[:, 1])
        plt.subplot(3, 4, 7)
        plt.scatter(outputs['sampling_grid'][1, :, 0], outputs['sampling_grid'][1, :, 1], marker='+')
        plt.subplot(3, 4, 8)
        plt.imshow(rectified_images_[1].astype(np.uint8))

        plt.subplot(3, 4, 9)
        plt.scatter(test_input_ctrl_pts[2, :, 0], test_input_ctrl_pts[2, :, 1])
        plt.subplot(3, 4, 10)
        plt.scatter(output_ctrl_pts[:, 0], output_ctrl_pts[:, 1])
        plt.subplot(3, 4, 11)
        plt.scatter(outputs['sampling_grid'][2, :, 0], outputs['sampling_grid'][2, :, 1], marker='+')
        plt.subplot(3, 4, 12)
        plt.imshow(rectified_images_[2].astype(np.uint8))

        plt.show()


if __name__ == '__main__':
    test_batch_transform()
