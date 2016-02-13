import matplotlib.pyplot as plt
from matplotlib import widgets


class DiscreteSlider(widgets.Slider):

    """A matplotlib slider widget with discrete steps.
    http://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete
    """

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.stepsize = kwargs.pop('stepsize', 1)
        widgets.Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = (int((val - self.valinit) / self.stepsize)
                        * self.stepsize + self.valinit)
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        # NOTE: I changed this
        # self.val = val
        self.val = discrete_val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)


def interactive_image(image_fn,
                      **args_dict):
    """
    http://matplotlib.org/examples/widgets/slider_demo.html
    """
    fig, ax = plt.subplots()

    plt.subplots_adjust(
        # left=0.25,
        # leave space at the bottom for the sliders
        bottom=0.1 + 0.05 * len(args_dict)
    )

    axes_image = plt.imshow(
        image_fn(**{k: v['default'] for k, v in args_dict.items()}))

    all_widgets = {}

    slider_bg_color = 'lightgoldenrodyellow'

    def widget_values():
        return {k: v.val for k, v in all_widgets.items()}

    def update(_):
        axes_image.set_array(
            image_fn(**widget_values()))
        fig.canvas.draw_idle()

    for idx, (arg, value_map) in enumerate(args_dict.items()):
        if value_map['widget'] == "slider":
            constructor_args = dict(
                ax=plt.axes([0.25, 0.05 + 0.05 * idx, 0.65, 0.03],
                            axisbg=slider_bg_color),
                label=value_map.get("label", arg),
                valinit=value_map['default'],
                valmin=value_map['lower'],
                valmax=value_map['upper'],
            )
            if "step" in value_map:
                constructor = DiscreteSlider
                constructor_args['stepsize'] = value_map['step']
            else:
                constructor = widgets.Slider
            slider = constructor(**constructor_args)
            slider.on_changed(update)
            all_widgets[arg] = slider
        else:
            raise ValueError("Unknown widget: %s" % value_map['widget'])

    plt.show()
    return widget_values()


if __name__ == "__main__":
    import numpy as np
    import scipy.misc

    foo = dict(
        widget="slider",
        default=0,
        lower=0.,
        upper=1.,
    )
    bar = dict(
        widget="slider",
        default=1.0,
        lower=0,
        upper=1.0,
    )

    img = scipy.misc.lena() / 255.

    def image_fn(foo, bar):
        tmp = np.clip(img, foo, bar)
        return tmp

    interactive_image(image_fn,
                      foo=foo,
                      bar=bar)
