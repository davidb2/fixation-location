'''Figure style of the paper and helpers to save panels and assemble figures.

Text is set in Times through LaTeX. Colors are tab10 with the tab20 light tints. Panels and figures are saved at 600 dpi.
'''
import io

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, ArrowStyle
from matplotlib.transforms import Bbox
from PIL import Image
from PIL.PngImagePlugin import PngInfo

DPI = 600

TAB10 = [tuple(c) for c in plt.get_cmap('tab10').colors]
TAB20 = [tuple(c) for c in plt.get_cmap('tab20').colors]
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GRAY, OLIVE, CYAN = TAB10
LBLUE, LORANGE, LGREEN, LRED, LPURPLE, LBROWN, LPINK, LGRAY, LOLIVE, LCYAN = TAB20[1::2]
INK = (0.15, 0.15, 0.15)

MUTANT, WILD, ORIGIN = BLUE, (1.0, 1.0, 1.0), GREEN
LINK = GRAY


def seq(c_light, c_dark, name):
  '''Sequential map from white through the tab20 light tint to the tab10 color.'''
  return LinearSegmentedColormap.from_list(name, [(1, 1, 1), c_light, c_dark])


CM_LAST = seq(LRED, RED, 'last')
CM_TEMP = seq(LPURPLE, PURPLE, 'temp')
CM_WILD = seq(LORANGE, ORANGE, 'wild')
CM_CYC = LinearSegmentedColormap.from_list('cyc', ['#9ad097', '#6bbe7c', '#41a65c', '#2c844c', '#355c40', '#354839'])  # location on the directed cycle, Figure 4a-d

RC = {
  'text.usetex': True,
  'text.latex.preamble': r'\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}',
  'font.family': 'serif',
  'font.size': 7.5,
  'axes.labelsize': 8,
  'axes.titlesize': 8,
  'xtick.labelsize': 7,
  'ytick.labelsize': 7,
  'legend.fontsize': 7,
  'axes.linewidth': 0.6,
  'axes.edgecolor': INK,
  'axes.labelcolor': INK,
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.labelpad': 2.5,
  'xtick.color': INK,
  'ytick.color': INK,
  'xtick.direction': 'out',
  'ytick.direction': 'out',
  'xtick.major.size': 3.0,
  'ytick.major.size': 3.0,
  'xtick.minor.size': 1.6,
  'ytick.minor.size': 1.6,
  'xtick.major.width': 0.6,
  'ytick.major.width': 0.6,
  'xtick.minor.width': 0.5,
  'ytick.minor.width': 0.5,
  'xtick.major.pad': 2.0,
  'ytick.major.pad': 2.0,
  'lines.linewidth': 1.3,
  'lines.solid_capstyle': 'butt',
  'patch.linewidth': 0.6,
  'legend.frameon': True,
  'legend.fancybox': False,
  'legend.framealpha': 1.0,
  'legend.edgecolor': '0.72',
  'legend.facecolor': 'white',
  'legend.borderpad': 0.45,
  'legend.handlelength': 1.8,
  'legend.handletextpad': 0.5,
  'legend.borderaxespad': 0.6,
  'savefig.facecolor': 'white',
  'figure.facecolor': 'white',
  'image.interpolation': 'nearest',
}
matplotlib.rcParams.update(RC)


def letter(fig, x, y, s, size=11):
  '''Bold serif panel letter, in figure coordinates of the top left corner.'''
  fig.text(x, y, r'\textbf{%s}' % s, fontsize=size, ha='left', va='top', color='black')


def legend(ax, **kw):
  lg = ax.legend(**kw)
  lg.get_frame().set_linewidth(0.5)
  return lg


def colorbar(fig, cax, cmap, vmin, vmax, ticks, label, orientation='vertical', extend='neither'):
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
  cb = fig.colorbar(sm, cax=cax, orientation=orientation, extend=extend, extendfrac=0.06)
  cb.outline.set_linewidth(0.5)
  cb.outline.set_edgecolor(INK)
  cb.set_ticks(ticks)
  cb.ax.tick_params(length=2.2, width=0.5, pad=1.5)
  cb.set_label(label, labelpad=2.5)
  return cb


HEAD = ArrowStyle('-|>', head_length=0.62, head_width=0.21)


def arrow(ax, p, q, color=LINK, lw=0.8, scale=7.0, zorder=2):
  '''Straight arrow from p to q with a sharp mitered head; p and q in data coordinates.'''
  a = FancyArrowPatch(
    p,
    q,
    arrowstyle=HEAD,
    mutation_scale=scale,
    lw=lw,
    color=color,
    shrinkA=0,
    shrinkB=0,
    joinstyle='miter',
    capstyle='butt',
    zorder=zorder,
  )
  ax.add_patch(a)
  return a


POLY = {3: (1.27, 0.0), 4: (1.27, np.pi / 4), 5: (1.06, 0.0), 6: (1.0, 0.0)}
POLY_MARKER = {3: '^', 4: 's', 5: 'p', 6: 'h'}


def poly_points(xy, n, R):
  k, orient = POLY[n]
  ang = np.pi / 2 + orient + 2 * np.pi * np.arange(n) / n
  pts = np.c_[np.cos(ang), np.sin(ang)] * k * R
  pts[:, 0] -= (pts[:, 0].max() + pts[:, 0].min()) / 2
  pts[:, 1] -= (pts[:, 1].max() + pts[:, 1].min()) / 2
  return pts + np.asarray(xy, float)


def polygon(ax, xy, n, R, fill, edgecolor=INK, lw=0.55, zorder=3):
  '''Regular polygon with n corners, centered on its bounding box at xy.'''
  from matplotlib.patches import Polygon

  p = Polygon(
    poly_points(xy, n, R),
    closed=True,
    facecolor=fill,
    edgecolor=edgecolor,
    lw=lw,
    zorder=zorder,
    joinstyle='miter',
  )
  ax.add_patch(p)
  return p


def _reach(n, R, phi):
  '''Distance from the center of a vertex to its boundary in the direction phi (disc if n is None).'''
  if n is None:
    return R
  pts = poly_points((0.0, 0.0), n, R)
  d = np.array([np.cos(phi), np.sin(phi)])
  best = np.inf
  for a, b in zip(pts, np.roll(pts, -1, axis=0)):
    e = b - a
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-12:
      continue
    t = (a[0] * e[1] - a[1] * e[0]) / den
    u = (a[0] * d[1] - a[1] * d[0]) / den
    if t > 0 and -1e-9 <= u <= 1 + 1e-9:
      best = min(best, t)
  return best


def draw_graph(
  ax,
  pos,
  links,
  fills,
  R=0.13,
  one_way=(),
  link_lw=0.9,
  node_lw=0.55,
  edgecolor=INK,
  square=(),
  gap=0.035,
  link_color=LINK,
  zorder=3,
  shapes=None,
):
  '''Discs of radius R. links are two-way pairs, one_way pairs end in an arrowhead, shapes maps vertices to corners.'''
  ax.set_aspect('equal')
  shp = dict(shapes or {})
  for v in square:
    shp.setdefault(v, 4)

  def ends(u, v, extra):
    p, q = np.array(pos[u], float), np.array(pos[v], float)
    d = (q - p) / np.linalg.norm(q - p)
    phi = np.arctan2(d[1], d[0])
    return p + d * _reach(shp.get(u), R, phi) * 0.95, q - d * (_reach(shp.get(v), R, phi + np.pi) + extra)

  for u, v in links:
    a, b = ends(u, v, -0.05 * R)
    ax.plot(*zip(a, b), color=link_color, lw=link_lw, solid_capstyle='butt', zorder=zorder - 1)
  for u, v in one_way:
    a, b = ends(u, v, gap)
    arrow(ax, a, b, color=link_color, lw=link_lw, zorder=zorder - 1)
  for v, (x, y) in pos.items():
    if v in shp:
      polygon(ax, (x, y), shp[v], R, fills[v], edgecolor=edgecolor, lw=node_lw, zorder=zorder)
    else:
      ax.add_patch(
        Circle((x, y), R, facecolor=fills[v], edgecolor=edgecolor, lw=node_lw, zorder=zorder)
      )
  ax.set_axis_off()


def boxed(ax, x, y, s, transform=None, size=8, ha='left', va='top'):
  '''A label in a thin square box, for example the fitness of a panel.'''
  kw = dict(
    fontsize=size,
    ha=ha,
    va=va,
    zorder=8,
    bbox=dict(boxstyle='square,pad=0.28', facecolor='white', edgecolor=INK, lw=0.5),
  )
  if transform is not None:
    kw['transform'] = transform
  return ax.text(x, y, s, **kw)


def shade(cmap, vals, vmin, vmax):
  return {v: cmap(np.clip((x - vmin) / (vmax - vmin), 0, 1)) for v, x in vals.items()}


def save_panel(fig, name, pad=0.03, box=None):
  '''Save name.png and name.pdf cropped to the panel, or to box = (x0, y0, x1, y1) in figure fractions.

  The PNG stores the pixel offset of the crop in the figure, which assemble uses.
  '''
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  boxes = [a.get_tightbbox(r) for a in fig.axes] + [t.get_window_extent(r) for t in fig.texts]
  boxes += [a.get_window_extent(r) for a in fig.artists] + [lg.get_window_extent(r) for lg in fig.legends]
  bb = Bbox.union([b for b in boxes if b is not None and b.width > 0])
  bb = bb.transformed(fig.dpi_scale_trans.inverted())
  w, h = fig.get_size_inches()
  x0 = max(0, int(np.floor((bb.x0 - pad) * DPI)))
  y0 = max(0, int(np.floor((bb.y0 - pad) * DPI)))
  x1 = min(int(round(w * DPI)), int(np.ceil((bb.x1 + pad) * DPI)))
  y1 = min(int(round(h * DPI)), int(np.ceil((bb.y1 + pad) * DPI)))
  if box is not None:
    x0, y0 = int(round(box[0] * w * DPI)), int(round(box[1] * h * DPI))
    x1, y1 = int(round(box[2] * w * DPI)), int(round(box[3] * h * DPI))
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=DPI, transparent=True)
  im = Image.open(buf)
  meta = PngInfo()
  meta.add_text('Offset', f'{x0},{y0}')
  im.crop((x0, im.height - y1, x1, im.height - y0)).save(name + '.png', pnginfo=meta, dpi=(DPI, DPI))
  fig.savefig(name + '.pdf', dpi=DPI, bbox_inches=Bbox([[x0 / DPI, y0 / DPI], [x1 / DPI, y1 / DPI]]))
  plt.close(fig)
  return show(name + '.png')


def assemble(size, panels):
  '''A figure of the given size in inches with every panel PNG placed at its stored pixel offset.'''
  fig = plt.figure(figsize=size, dpi=DPI)
  for path in panels:
    im = Image.open(path)
    x0, y0 = (int(v) for v in im.info['Offset'].split(','))
    fig.figimage(np.asarray(im), xo=x0, yo=y0, origin='upper', zorder=1)
  return fig


def save_figure(fig, name):
  '''Save name.png at 600 dpi.'''
  fig.savefig(name + '.png', dpi=DPI)
  plt.close(fig)
  return show(name + '.png')


def show(path, width=1400):
  '''A reduced copy of a saved PNG for display in a notebook.'''
  from IPython.display import Image as Shown

  im = Image.open(path).convert('RGBA')
  bg = Image.new('RGBA', im.size, 'white')
  im = Image.alpha_composite(bg, im).convert('RGB')
  if im.width > width:
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
  buf = io.BytesIO()
  im.save(buf, format='PNG')
  return Shown(data=buf.getvalue())
