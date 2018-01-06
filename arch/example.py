# pylint: disable=C,R,E1101
'''
Minimalist example of usage of SE(3) CNN

This example train a neural network to classify two geometrical objects
A ball : x^2 + y^2 + z^2 < radius^2
An octahedron : |x| + |y| + |z| < radius

The input data is generated on the fly, the radius is random and noise is added
'''
import torch
import numpy as np

# The Highway Block is a way to introduce non linearity into the neural network
# The class HighwayBlock inherit from the class torch.nn.Module.
# It contains one convolution, some ReLU and multiplications
from se3_cnn.blocks.highway import HighwayBlock


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # The parameters of a HighwayBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - The stride, same as 2D convolution
        # - A parameter to tell if the non linearity is enabled or not (ReLU or nothing)
        features = [
            (1, ),  # As input we have a scalar field
            (2, 1, 0, 3, 3),  # Note that this particular choice of multiplicities it completely arbitrary
            (0, 1, 0, 2),
            (2, )  # Two scalar fields as output
        ]
        common_block_params = {'size': 3, 'radial_amount': 2, 'stride': 2, }
        block_params = [
            {'non_linearities': True},
            {'non_linearities': True},
            {'non_linearities': False},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [HighwayBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''

        # Perform the Highway blocks
        x = self.blocks(inp)  # [batch, features, x, y, z]

        # Take the average over the spacial indicies
        x = x.view(x.size(0), x.size(1), -1)  # [batch, features, x*y*z]
        x = x.mean(-1)  # [batch, features]

        return x


def main():
    cnn = CNN()

    if torch.cuda.is_available():
        cnn.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-2)

    batch_size = 64
    sample_size = 24  # Size of the input cube

    mesh = np.linspace(-1, 1, sample_size)
    mx, my, mz = np.meshgrid(mesh, mesh, mesh)

    def step(i):
        # only noise
        x = 0.1 * np.random.randn(batch_size, 1, sample_size, sample_size, sample_size)
        # random labels: zero for the ball and one for the octahedron
        y = np.random.randint(0, 2, size=(batch_size,))

        # add the balls and octahedrons on top of the noise
        for j, label in enumerate(y):
            radius = 0.6 + np.random.rand() * (0.9 - 0.6)

            if label == 0:
                # ball
                mask = mx ** 2 + my ** 2 + mz ** 2 < radius ** 2
            if label == 1:
                # octahedron
                mask = abs(mx) + abs(my) + abs(mz) < radius

            x[j, 0, mask] += np.random.randint(2) * 2 - 1

        # convert the input and labels into Pytorch tensor
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        # upload on the GPU if available
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        # forward and backward propagation
        optimizer.zero_grad()
        out = cnn(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # download results on the CPU
        loss = loss.data.cpu().numpy()
        out = out.data.cpu().numpy()
        y = y.data.cpu().numpy()

        # compute the accuracy
        acc = np.sum(out.argmax(-1) == y) / batch_size

        print("{}: acc={}% loss={}".format(i, 100 * acc, float(loss)))

    for i in range(1000):
        step(i)


if __name__ == '__main__':
    main()
