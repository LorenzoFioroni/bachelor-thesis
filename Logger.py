import os
import time

# Utility class to log messages to terminal and to file.
class Logger:
    def __init__(self, toTerminal = True, toFile = True):
        self.toTerminal = toTerminal    # Default choice for logging to terminal
        self.toFile = toFile            # Default choice for logging to file

        self.folderName = "{}".format(time.strftime("%Y-%m-%d_%H:%M:%S"))
        self.folderPath = "{}/logs/{}".format(os.getcwd(), self.folderName)

        # Creates the directory structure
        try:
            os.makedirs("{}/{}".format(self.folderPath, "plots"))
            self.fileName = "trace.log"
            self.outFile = open("{}/{}".format(self.folderPath,self.fileName), "x")
            self.log("Logging on {}/{}\n".format(self.folderPath,self.fileName),toFile = False)
        except OSError:
            raise Exception ("Creation of the directory structure failed")


    # Function to log to terminal and to file. If toTerminal or toFile is not 
    # specified, the default option provided when instantiating is used.
    # The default option is to add the timestamp to each printed line but it 
    # can be turned off setting timestamp=False
    def log(self, msg, toTerminal = None, toFile = None, timestamp = True):
        if toTerminal == None: toTerminal = self.toTerminal
        if toFile == None: toFile = self.toFile

        if timestamp: msg = "{}\t{}".format(time.strftime("%X"), msg)

        if toTerminal: print(msg)

        if toFile and self.outFile:
            print(msg, file=self.outFile)


    # Function to close the file when no longer needed
    def close(self):
        if self.outFile:
            self.outFile.close()


    # Utility function to print an horizontal line (without timestamp)
    def separator(self, toTerminal = None, toFile = None):
        if toTerminal == None: toTerminal = self.toTerminal
        if toFile == None: toFile = self.toFile
        self.log("\n{}\n".format("-"*30), toTerminal, toFile, timestamp = False)


    # Utility function to save the last matplotlib figure to file in the same 
    # folder as trace.log
    def savePlot(self, plt, name):
        plt.savefig("{}/plots/{}".format(self.folderPath, name), dpi=600)
