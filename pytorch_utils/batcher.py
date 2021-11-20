import re

class batcher(object):
    '''
        Wrapper to simplify tracking scalars during batch training.

        Useful to track epoch averages, for example.
    '''
    def __init__(self):
        self.Nbatch = 0
        self.vars = {}

    def batch(self):
        '''
            Call at the beginning of each batch
        '''
        self.Nbatch = self.Nbatch + 1

    def add(self,var,val):
        '''
            Add value 'val' to scalar 'var'. If 'var' is not yet tracked
            set val = var

            :param val (String): scalar variable to add var to
            :param var (scalar): value to add to var
        '''
        try:
            self.vars[var] = self.vars[var] + val

        except KeyError:
            self.vars[var] = val

    def report(self,rexp=None):
        '''
            Print out average values of tracked scalars since last reset()

            :param rexp (String): only print scalars with names matching
                the regular expression rexp
        '''

        if rexp is not None:
            p = re.compile(rexp)

        for key, val in self.vars.items():
            if rexp is not None:
                if not p.match(key):
                    continue

            print(key + ": %.4e, " % (val / self.Nbatch,), end='')

    def write(self,summary_writer,epoch):
        '''
            Add average of scalars since last reset() to tensorboardX
            SummaryWriter

            calls reset() after write

            :param (tensorboardX.SummaryWriter): SummaryWriter
            :param epoch (scalar): global_idx to pass to SummaryWriter
        '''
        for key, val in self.vars.items():
            summary_writer.add_scalar(key,val / self.Nbatch,epoch)
        self.reset()

    def reset(self):
        '''
            Delete all tracked scalars and reset batch count
        '''
        self.vars = {}
        self.Nbatch = 0
