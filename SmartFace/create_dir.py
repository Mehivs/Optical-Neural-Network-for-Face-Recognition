import os
import shutil

class create(object):
    def path(path):
        try:
            shutil.rmtree(path)
        except OSError:
            print('path not exist, creating ...')
            pass

        if not os.path.exists(path):
            os.mkdir(path)
        return
