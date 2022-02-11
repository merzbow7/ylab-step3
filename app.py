from flask import Flask, request, render_template

from geometry import figure

app = Flask("__file__")

geometry_classes = {f'{cls}': '-'.join(getattr(figure, cls).__slots__)
                    for cls in figure.__all__}
print(geometry_classes)


@app.route('/')
def index():
    print(request.args)
    cls_name = request.args.get('figure', default='Circle', type=str)
    figure_class = getattr(figure, cls_name)
    class_args = {attr: request.args.get(attr, default=1, type=float)
                  for attr in figure_class.__slots__}
    print(f"{cls_name=}")
    print(f"{class_args=}")
    shape = figure_class(**class_args)
    img = shape.plot()
    slots = {slot: class_args[slot] for slot in shape.__slots__}
    return render_template('index.html',
                           img=img,
                           slots=slots,
                           choose=cls_name,
                           figures=figure.__all__)


if __name__ == '__main__':
    app.run()
