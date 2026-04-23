import sys, argparse
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from splice import _LocWrapper

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--encoder', default='satclip', help='Location encoder name, used in output filename')
args = parser.parse_args()
encoder = args.encoder

# ── Load model ───────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
splice_model = torch.load('../splice_results/splice_model.pt', map_location=device, weights_only=False)
concepts_list = torch.load('../splice_results/concepts.pt', map_location=device, weights_only=False)
splice_model.eval()

def get_top(lat, lon, k=4):
    ll = torch.tensor([[lat, lon]], dtype=torch.float32, device=device)
    with torch.no_grad():
        w, _ = splice_model.encode_image(ll)
    pairs = sorted([(c, v) for c, v in zip(concepts_list, w[0].tolist()) if v > 0],
                   key=lambda x: -x[1])
    seen, out = set(), []
    for c, v in pairs:
        if c not in seen:
            seen.add(c); out.append((c, v))
    return out[:k]

# ── Places ───────────────────────────────────────────────────────────────────
PLACES = [
    ('Amazon\nRainforest',    -3.47,  -60.02, 'left'),
    ('Sahara\nDesert',        25.00,   10.00, 'left'),
    ('Patagonian\nGlaciers', -50.50,  -73.00, 'left'),
    ('Grand Canyon',          36.10, -112.10, 'left'),
    ('Iceland',               64.50,  -17.50, 'left'),
    ('Alps',                  46.50,    8.00, 'right'),
    ('Nile Delta',            31.30,   30.50, 'right'),
    ('Sundarbans',            21.90,   89.20, 'right'),
    ('Tibetan\nPlateau',      32.00,   90.00, 'right'),
    ('Siberian\nTaiga',       62.00,  105.00, 'right'),
    ('Mekong\nDelta',         10.10,  105.70, 'right'),
]

LOC_COLORS = [
    '#2E86AB',  # Amazon
    '#E07B39',  # Sahara
    '#5C6BC0',  # Patagonian Glaciers
    '#C0392B',  # Grand Canyon
    '#27AE60',  # Iceland
    '#8E44AD',  # Alps
    '#D4AC0D',  # Nile Delta
    '#1ABC9C',  # Sundarbans
    '#E74C3C',  # Tibetan Plateau
    '#2980B9',  # Siberian Taiga
    '#16A085',  # Mekong Delta
]

decomps = {}
for name, lat, lon, _ in PLACES:
    decomps[name] = get_top(lat, lon, k=4)
    print(f'{name.replace(chr(10)," "):24s}: {[(c,round(v,3)) for c,v in decomps[name]]}')

# ── Style ────────────────────────────────────────────────────────────────────
FIG_BG    = '#FFFFFF'
PANEL_BG  = '#FFFFFF'
TEXT_DARK = '#111111'
TEXT_MED  = '#555555'

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'axes.linewidth': 0.7,
})

# ── Figure ───────────────────────────────────────────────────────────────────
PANEL_H, PANEL_GAP = 0.135, 0.020
left_m, right_m, bot_m, top_m = 0.21, 0.21, 0.03, 0.03
panel_w = left_m - 0.014

fig = plt.figure(figsize=(22, 11.5), facecolor=FIG_BG)

ax_map = fig.add_axes(
    [left_m, bot_m, 1 - left_m - right_m, 1 - bot_m - top_m],
    projection=ccrs.Robinson()
)
ax_map.set_global()
ax_map.stock_img()
ax_map.add_feature(
    cfeature.NaturalEarthFeature('physical', 'coastline', '110m'),
    facecolor='none', edgecolor='#00000055', linewidth=0.5, zorder=3
)
ax_map.add_feature(
    cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '110m'),
    facecolor='none', edgecolor='#00000033', linewidth=0.3, zorder=3
)
ax_map.gridlines(color='#00000012', linewidth=0.4, linestyle='-', zorder=2)

# ── Panel helpers ─────────────────────────────────────────────────────────────
def vert_positions(n):
    block = n * PANEL_H + (n - 1) * PANEL_GAP
    start = bot_m + ((1 - top_m - bot_m) - block) / 2
    return [start + i * (PANEL_H + PANEL_GAP) for i in range(n)][::-1]

def fmt_latlon(lat, lon):
    la = f'{abs(lat):.2f}°{"N" if lat >= 0 else "S"}'
    lo = f'{abs(lon):.2f}°{"E" if lon >= 0 else "W"}'
    return f'{la}, {lo}'

def draw_panel(fig_x, fig_y, fig_w, fig_h, name, decomp, color, lat, lon):
    ax = fig.add_axes([fig_x, fig_y, fig_w, fig_h])
    ax.set_facecolor(PANEL_BG)
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(color)
    ax.spines['left'].set_linewidth(3.5)

    labels = [c for c, _ in decomp]
    values = [w for _, w in decomp]
    y_pos  = np.arange(len(labels))
    max_v  = max(values) if values else 1
    alphas = [0.90, 0.65, 0.45, 0.30]

    for i, (y, val) in enumerate(zip(y_pos, values)):
        ax.barh(y, val, color=color, height=0.58, zorder=2,
                alpha=alphas[i] if i < len(alphas) else 0.25)
        ax.text(val + max_v * 0.03, y, f'{val:.3f}',
                va='center', ha='left', fontsize=8.5, color=TEXT_MED)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_DARK)
    ax.invert_yaxis()
    ax.set_xlim(0, max_v * 1.55)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.set_title(f'{name.replace(chr(10), " ")}  {fmt_latlon(lat, lon)}',
                 fontsize=10.5, fontweight='bold', color=color, pad=6, loc='left')
    ax.grid(False)

left_places  = [(n, la, lo, s) for n, la, lo, s in PLACES if s == 'left']
right_places = [(n, la, lo, s) for n, la, lo, s in PLACES if s == 'right']
left_ys  = vert_positions(len(left_places))
right_ys = vert_positions(len(right_places))

line_specs = []

for idx, ((name, lat, lon, side), py) in enumerate(
        zip(left_places + right_places, left_ys + right_ys)):
    color   = LOC_COLORS[idx]
    is_left = side == 'left'
    fig_x   = 0.004 if is_left else (1 - panel_w + 0.010)

    draw_panel(fig_x, py, panel_w, PANEL_H, name, decomps[name], color, lat, lon)

    panel_conn_x = fig_x + panel_w if is_left else fig_x
    panel_conn_y = py + PANEL_H / 2

    ax_map.plot(lon, lat, 'o', markersize=7.5, color=color,
                markeredgecolor='white', markeredgewidth=1.1,
                transform=ccrs.PlateCarree(), zorder=6)

    line_specs.append((panel_conn_x, panel_conn_y, lon, lat, color))

# Force layout so cartopy transforms are initialised
fig.canvas.draw()

for panel_conn_x, panel_conn_y, lon, lat, color in line_specs:
    proj_pt = ax_map.projection.transform_point(lon, lat, ccrs.PlateCarree())
    disp    = ax_map.transData.transform(proj_pt)
    map_x, map_y = fig.transFigure.inverted().transform(disp)

    fig.add_artist(Line2D(
        [panel_conn_x, map_x], [panel_conn_y, map_y],
        transform=fig.transFigure,
        color=color, linewidth=1.5, linestyle=(0, (5, 3)),
        alpha=0.85, zorder=10, clip_on=False
    ))

out_stem = f'splice_world_{encoder}'
plt.savefig(f'{out_stem}.pdf', dpi=300, bbox_inches='tight', facecolor=FIG_BG)
plt.savefig(f'{out_stem}.png', dpi=200, bbox_inches='tight', facecolor=FIG_BG)
print(f'Saved {out_stem}.pdf and {out_stem}.png')
