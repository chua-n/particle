"""Render the mayavi scene off screen.

When you don't need display the mayavi scene while running python scripts, 
importing `mlab` from this module is recommended, i.e., 

from mayaviOffScreen imoprt mlab

Notes:
-------
    1. If you are using mayavi from remote host by SSH, etc, you must 
        import `mlab` from this module. Or something wrong happens!
    1. Under the circumstance above, if you need import some other modules meanwhile
        and those modules import `mlab` as well in themselves, make sure the sentence
        `from mayaviOffScreen import mlab` appears before `import otherModule`!
        So that `mayaviOffScreen.mlab` covers `otherModule.mlab`.
"""
from pyvirtualdisplay import Display

display = Display(visible=False, size=(1280, 1024))
display.start()

from mayavi import mlab
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
