import time
import os
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from collections import defaultdict
import itertools


class Logger:
    def __init__(self, args):
        self.args = args

        self.train_losses = pd.DataFrame()
        self.train_losses_idx = 0

        self.test_losses = pd.DataFrame()
        self.test_losses_idx = 0

        self.val_losses = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

        self.create_log_path(args)

    def create_log_path(self, args):

        args.log_path = os.path.join(args.save_folder, args.task, args.time)

        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)

        self.log_file = os.path.join(args.log_path, "log.txt")

        self.write_to_log_file(args)

        args.imp_file = os.path.join(args.log_path, "imp.pt")
        args.gnn_file = os.path.join(args.log_path, "gnn.pt")
        args.optimizer_file = os.path.join(args.log_path, "optimizer.pt")

        args.plotdir = os.path.join(args.log_path, "plots")
        if not os.path.exists(args.plotdir):
            os.makedirs(args.plotdir)

    def save_checkpoint(self, args, imp, gnn, optimizer, specifier=""):
        args.imp_file = os.path.join(args.log_path, "imp_{}.pt".format(specifier))
        args.gnn_file = os.path.join(args.log_path, "gnn_{}.pt".format(specifier))
        args.optimizer_file = os.path.join(
            args.log_path, "optimizer_{}.pt".format(specifier)
        )

        if imp is not None:
            torch.save(imp.state_dict(), args.imp_file)
        if gnn is not None:
            torch.save(gnn.state_dict(), args.gnn_file)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), args.optimizer_file)

    def write_to_log_file(self, string):
        """
        Write given string in log-file and print as terminal output
        """
        print(string)
        cur_file = open(self.log_file, "a")
        print(string, file=cur_file)
        cur_file.close()

    def create_log(
        self,
        args,
        specifier,
        imp=None,
        gnn=None,
        accuracy=None,
        optimizer=None,
    ):

        print("Saving model and log-file to " + args.log_path)

        # Save losses throughout training and plot
        self.train_losses.to_pickle(os.path.join(self.args.log_path, "train_loss"))

        if self.val_losses is not None:
            self.val_losses.to_pickle(os.path.join(self.args.log_path, "val_loss"))

        if accuracy is not None:
            np.save(os.path.join(self.args.log_path, "accuracy"), accuracy)

        # Save the model checkpoint
        self.save_checkpoint(args, imp, gnn, optimizer, specifier=specifier)

    def draw_loss_curves(self):
        for i in self.train_losses.columns:
            plt.figure()
            plt.plot(self.train_losses[i], "-b", label="train " + i)

            if self.val_losses is not None and i in self.val_losses:
                plt.plot(self.val_losses[i], "-r", label="val " + i)

            plt.xticks(np.arange(0,len(self.train_losses[i]),10))
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")

            # save image
            plt.savefig(os.path.join(self.args.log_path, i + ".png"))
            plt.close()

    def append_train_loss(self, loss):
        for k, v in loss.items():
            self.train_losses.at[str(self.train_losses_idx), k] = np.mean(v)
        self.train_losses_idx += 1

    def append_val_loss(self, val_loss):
        for k, v in val_loss.items():
            self.val_losses.at[str(self.val_losses_idx), k] = np.mean(v)
        self.val_losses_idx += 1

    def append_test_loss(self, test_loss):
        for k, v in test_loss.items():
            if type(v) != defaultdict:
                self.test_losses.at[str(self.test_losses_idx), k] = np.mean(v)
        self.test_losses_idx += 1

    def result_string(self, trainvaltest, epoch, losses, t=None):
        string = ""
        if trainvaltest == "test":
            string += (
                "-------------------------------- \n"
                "--------Testing----------------- \n"
                "-------------------------------- \n"
            )
        else:
            string += str(epoch) + " " + trainvaltest + "\t \t"

        for loss, value in losses.items():
            if type(value) == defaultdict:
                string += loss + " "
                for idx, elem in sorted(value.items()):
                    string += str(idx) + ": {:.10f} \t".format(
                        np.mean(list(itertools.chain.from_iterable(elem)))
                    )
            elif np.mean(value) != 0 and not math.isnan(np.mean(value)):
                string += loss + " {:.10f} \t".format(np.mean(value))

        if t is not None:
            string += "time: {:.4f}s \t".format(time.time() - t)

        return string
    
    def finetune_result(self, args, test_acc_list, val_acc_list):
        args.log_result_path = os.path.join(args.save_folder, args.task, 'result.log')
        with open(args.log_result_path, 'a+') as f:
            f.write(args.dataset + ' ' + str(args.runseed) + ' ' + str(np.array(test_acc_list).max()))
            f.write('  test of best_val: ' + str(np.array(test_acc_list)[np.array(val_acc_list) == np.array(val_acc_list).max()][-1]) + '  last_epoch: ' + str(np.array(test_acc_list[-1])))
            #f.write(args.input_model_file + ' ' +str(args.dropout_ratio))
            f.write('\n')
