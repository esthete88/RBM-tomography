from IPython.display import clear_output
import matplotlib.pyplot as plt


class PrintCallback:
    """Simple callback that prints current loss and plots loss history with moving average.

    Parameters
    ----------
    freq : int
        How often output is displayed.
    plot : bool
        If `True`, plots loss history.
    gamma : float
        Moving average parameter.
    """
    def __init__(self, freq=100, plot=False, gamma=0.99):
        self.loss_history = []
        self.ma_history = []
        self.freq = freq
        self.plot = plot
        self.gamma = gamma

    def __call__(self, model, epoch_log):
        epoch = epoch_log['epoch']
        n_epochs = epoch_log['n_epochs']
        loss = epoch_log['loss']
        
        if len(self.ma_history) != 0:
            ma = self.ma_history[-1] * self.gamma + loss * (1 - self.gamma)
        else:
            ma = loss
            
        self.loss_history.append(loss)
        self.ma_history.append(ma)

        if (epoch + 1) % self.freq == 0:
            clear_output(True)

            message_head = 'Epoch {}/{}\n'.format(epoch + 1, n_epochs)
            message_loss = '\tLoss: {:<10}'.format(loss)                
            print(message_head + message_loss)
            
            if 'sampled_indices' in epoch_log.keys():
                print('\tSampled indices:', epoch_log['sampled_indices'])

            if self.plot:
                plt.plot(self.loss_history, label='loss')
                plt.plot(self.ma_history, label='MA 100')
                plt.legend()
                plt.show()


class GibbsStepsIncreaseCallback:
    """Callback for updating number of steps in Gibbs sampling.
       Increases `n_gibbs_steps` linearly each `freq` epoch ending at `stop_epoch`."""
    def __init__(self, freq=500, init_epoch=0, stop_epoch=1500):
        self.freq = freq
        self.init_epoch = init_epoch
        self.stop_epoch = stop_epoch
        
    def __call__(self, model, epoch_log):
        epoch = epoch_log['epoch']
        
        if (epoch - self.init_epoch + 1) % self.freq == 0 and epoch < self.stop_epoch:
            model.n_gibbs_steps += 1


class TemperatureDecayCallback:
    """Callback for updating temperature of sampling.
       Decreases temperature linearly each `freq` epoch ending at `stop_epoch`."""
    def __init__(self, init_temperature=15, freq=1, init_epoch=0, stop_epoch=1000):
        self.freq = freq
        self.init_temperature = init_temperature
        self.init_epoch = init_epoch
        self.stop_epoch = stop_epoch
        
    def __call__(self, model, epoch_log):
        epoch = epoch_log['epoch']
        
        if epoch == self.init_epoch:
            model.temperature = self.init_temperature
        
        if (epoch - self.init_epoch + 1) % self.freq == 0 and epoch < self.stop_epoch:
            model.temperature -= (model.temperature - 1) / (self.stop_epoch - epoch)
