import numpy as np
import matplotlib.pyplot as plt

def makeComparisons(nrmse, samples, p, m, noise, step, appendVal, length, name=None):
    nrmse.insert(0, appendVal)
    x=np.arange(length)
    fig, axs = plt.subplots(3,2)

    im=axs[0,0].plot(np.arange(length+1),nrmse)
    axs[0,0].set_title('NRMSE') # End:', nrmse[-1])

    im=axs[1,0].plot(np.arange(length+1), samples)
    axs[1,0].set_title('Samples')

    im=axs[2,0].plot(x, p)
    axs[2,0].set_title('P Grad')

    im=axs[0,1].plot(x, m)
    axs[0,1].set_title('M Grad')

    im=axs[1,1].plot(x, noise)
    axs[1,1].set_title('Noise')

    im=axs[2,1].plot(x, step)
    axs[2,1].set_title('Step Size')
    fig.suptitle('Random Initialization-- start at 0 Iteration')
    fig.tight_layout()
    fig.savefig(f'Scaling.jpg')

def visualize(sample, ground_truth, difference_map, SSIM, nrmse_vals, ssim_vals, epochStart, sliceNum):
    fig, axs = plt.subplots(2,2)
    fig.tight_layout()
    im=axs[0, 0].imshow(np.abs(sample))
    axs[0, 0].set_title(f'Sample Reconstruction')
    fig.colorbar(im, ax=axs[0,0])

    im=axs[0, 1].imshow(np.abs(ground_truth))
    axs[0, 1].set_title(f'GROUND TRUTH')
    fig.colorbar(im, ax=axs[0,1])

    im=axs[1,0].imshow(np.abs(difference_map))
    bestRMSE="%.5f" % nrmse_vals[-1]
    axs[1,0].set_title(f'DIFFERENCE MAP, RMSE SCORE: {bestRMSE}')
    fig.colorbar(im, ax=axs[1,0])

    im=axs[1,1].imshow(np.abs(SSIM))
    bestSSIM="%.5f" % ssim_vals[-1]
    axs[1,1].set_title(f'SSIM SCORE: {bestSSIM}')
    fig.colorbar(im, ax=axs[1,1])
    fig.suptitle(f'Random init, Epoch Start: {epochStart}')
    fig.tight_layout()
    plt.tight_layout(h_pad=1.0)
    fig.savefig(f'groundTruthComparison_{sliceNum}.jpg')