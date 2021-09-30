import datetime as dt


class Logger:
    def __init__(self, log_fname):
        self.fname = './Logs/' + log_fname + '.txt'

    def log(self, log_str='', parameters=[]):
        prefix = '[{}]'.format(dt.datetime.now())
        if len(parameters) > 0:
            prefix = prefix + ' ---------- parameters: {}'.format(parameters)
        full_str = prefix + ' ' + log_str + '\n'
        print('Writing to log:\n{}'.format(full_str))
        with open(self.fname, 'a') as f:
            f.write(full_str)
