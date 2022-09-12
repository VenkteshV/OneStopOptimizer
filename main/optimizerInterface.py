from abc import ABCMeta, abstractstaticmethod
import abc
import zope.interface

class OptimizerInterface(zope.interface.Interface):
    def evaluation_function(self):
        pass
    def run_trials(self):
        pass