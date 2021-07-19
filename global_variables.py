import os
def init(dir, save_extra):
    global output_dir
    output_dir = dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    global save_extra_stuff
    save_extra_stuff = save_extra

def set_size(size_00, size_11):
    global size_0
    size_0 = size_00
    global size_1
    size_1 = size_11