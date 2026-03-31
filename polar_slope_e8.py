
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, sys, time
 
 
class E8Lattice:
    def __init__(self):
        self.roots = self._generate_roots()
        self.min_norm = np.sqrt(2)
        self.projected_slopes = self._compute_projection_slopes()
 
    def _generate_roots(self):
        roots = []
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1,1]:
                    for s2 in [-1,1]:
                        r = np.zeros(8); r[i]=s1; r[j]=s2; roots.append(r)
        for mask in range(256):
            signs = [1 if (mask>>i)&1 else -1 for i in range(8)]
            if sum(1 for s in signs if s==-1) % 2 == 0:
                roots.append(np.array([s*0.5 for s in signs]))
        return np.array(roots)
 
    def _compute_projection_slopes(self):
        slopes = []
        for root in self.roots:
            x, y = np.sum(root[:4]), np.sum(root[4:])
            slopes.append(y/x if abs(x) > 0.01 else np.sign(y)*10)
        return np.array(slopes)
 
    def assign_root(self, ng):
        phase = (np.sqrt(max(ng, 0.01)) / self.min_norm) % 1.0
        return int(phase * len(self.roots)) % len(self.roots)
 
 
def sieve(n):
    if n <= 0: return []
    limit = max(15, int(n*(math.log(n)+math.log(math.log(n))))+100)
    while True:
        ip = [True]*(limit+1); ip[0]=ip[1]=False
        for i in range(2, int(limit**0.5)+1):
            if ip[i]:
                for j in range(i*i, limit+1, i): ip[j]=False
        p = [i for i,v in enumerate(ip) if v]
        if len(p)>=n: return p[:n]
        limit = int(limit*1.5)+100
 
 
def main():
    n_primes = 20000
    outfile = '/mnt/user-data/outputs/e8_polar_slope_panels.png'
    if len(sys.argv)>=2:
        try: n_primes=int(sys.argv[1])
        except: pass
    if len(sys.argv)>=3: outfile=sys.argv[2]
 
    print(f"E8 Polar Slope Coloring — {n_primes:,} primes")
    t0 = time.time()
 
    primes = np.array(sieve(n_primes), dtype=np.float64)
    print(f"  Sieved {len(primes):,} primes up to {int(primes[-1]):,}")
 
    e8 = E8Lattice()
    gaps = np.diff(primes)
    logp = np.log(primes[:-1]); logp[logp<1]=1
    ng = gaps/logp
    ri = np.array([e8.assign_root(g) for g in ng])
    rs = e8.projected_slopes[ri]
    slopes = np.zeros(len(primes))
    slopes[1:] = np.clip(rs, -3, 3)
 
    cmap, vmin, vmax = 'coolwarm', -3, 3
 
    # === 4-panel figure ===
    fig, axes = plt.subplots(2, 2, figsize=(20,20), facecolor='black')
    for ax in axes.flat: ax.set_facecolor('black')
 
    configs = [
        (lambda p: p,        primes, 'p',     f'first {n_primes:,}',       0.4, 0, 0),
        (lambda p: np.sqrt(p), primes, '√p',  f'first {n_primes:,}',       0.5, 0, 1),
        (lambda p: np.sqrt(p), primes[:min(3000,n_primes)], '√p',
                                               f'first {min(3000,n_primes)} (detail)', 1.2, 1, 0),
        (lambda p: np.log(p),  primes, 'log p', f'first {n_primes:,}',     0.5, 1, 1),
    ]
 
    for rfunc, pdata, rlabel, subtitle, sz, row, col in configs:
        r = rfunc(pdata)
        x, y = r*np.cos(pdata), r*np.sin(pdata)
        sl = slopes[:len(pdata)]
        sc = axes[row,col].scatter(x, y, s=sz, c=sl, cmap=cmap,
                                    vmin=vmin, vmax=vmax, marker='.', linewidths=0, alpha=0.9)
        axes[row,col].set_aspect('equal'); axes[row,col].axis('off')
        axes[row,col].set_title(f'(r, θ) = ({rlabel}, p) — {subtitle}',
                                 color='white', fontsize=12, pad=10)
 
    fig.suptitle(f'Prime Polar Self-Map with E₈ Root Projection Slope\n'
                 f'{n_primes:,} primes — color = slope of assigned E₈ root projected to R²',
                 color='white', fontsize=16, y=0.98)
 
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('E₈ projection slope (y/x)', fontsize=11, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"  Panels → {outfile}")
    plt.close()
 
    # === Standalone √p version ===
    fig2, ax2 = plt.subplots(figsize=(16,16), facecolor='black')
    ax2.set_facecolor('black')
    r2 = np.sqrt(primes)
    x2, y2 = r2*np.cos(primes), r2*np.sin(primes)
    sc2 = ax2.scatter(x2, y2, s=0.6, c=slopes, cmap=cmap,
                       vmin=vmin, vmax=vmax, marker='.', linewidths=0, alpha=0.9)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title(f'Prime Polar Self-Map — E₈ Slope Coloring\n'
                   f'(r, θ) = (√p, p) — {n_primes:,} primes',
                   color='white', fontsize=15, pad=20)
    cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.7, aspect=30, pad=0.02)
    cb2.set_label('E₈ projection slope', fontsize=11, color='white')
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')
    ax2.text(0.5, -0.02,
             'Color = slope of E₈ root assigned via normalized prime gap.\n'
             'Red = steep positive, Blue = steep negative, White = near zero.',
             transform=ax2.transAxes, fontsize=9, ha='center', va='top', color='#aaaaaa')
    save2 = outfile.replace('.png','_sqrt.png')
    plt.savefig(save2, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"  Standalone √p → {save2}")
    plt.close()
 
    # === Standalone r=p version ===
    fig3, ax3 = plt.subplots(figsize=(16,16), facecolor='black')
    ax3.set_facecolor('black')
    x3, y3 = primes*np.cos(primes), primes*np.sin(primes)
    sc3 = ax3.scatter(x3, y3, s=0.4, c=slopes, cmap=cmap,
                       vmin=vmin, vmax=vmax, marker='.', linewidths=0, alpha=0.85)
    ax3.set_aspect('equal'); ax3.axis('off')
    ax3.set_title(f'Prime Polar Self-Map — E₈ Slope Coloring\n'
                   f'(r, θ) = (p, p) — {n_primes:,} primes',
                   color='white', fontsize=15, pad=20)
    cb3 = plt.colorbar(sc3, ax=ax3, shrink=0.7, aspect=30, pad=0.02)
    cb3.set_label('E₈ projection slope', fontsize=11, color='white')
    cb3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='white')
    save3 = outfile.replace('.png','_linear.png')
    plt.savefig(save3, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"  Standalone r=p → {save3}")
    plt.close()
 
    print(f"\nDone in {time.time()-t0:.1f}s")
 
 
if __name__ == "__main__":
    main()
