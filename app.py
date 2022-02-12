from flask import Flask, request, render_template

from geometry import figure

app = Flask("__file__")


@app.route('/')
def index() -> str:
    cls_name = request.args.get('figure', default='Circle', type=str)
    figure_class = getattr(figure, cls_name)
    slots = figure_class.__slots__
    class_args = {attr: request.args.get(attr, type=float, default=10)
                  for attr in slots}
    shape = figure_class(**class_args)
    img = shape.plot()
    summary = shape.get_summary()
    slots = {slot: class_args[slot] for slot in slots}
    return render_template('index.html',
                           img=img,
                           slots=slots,
                           choose=cls_name,
                           summary=summary,
                           figures=figure.__all__)


if __name__ == '__main__':
    app.run()
