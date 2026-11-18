"""Plot individual prior crossings and the new matched readout contrasts."""

from kv_route_contrast import configure
import kv_route_run as R


def main():
    configure()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prior = R.read("audit.json")["retrospective"]["rows"]
    transfer = R.OUT.with_name("kv_route_20261115_v3")/"matched-contrast.json"
    confirm = R.read("matched-contrast.json")["suites"] + R.Base.Previous.load(transfer)["suites"]
    fig, (left, right) = plt.subplots(1,2,figsize=(12.5,5.1),layout="constrained")
    cases = [(-r["original"], r["lift"], r["flip"]) for r in prior]
    for flip, color, tag in ((False,"#71859b","365 same-code cases"),(True,"#de695c","19 flips")):
        xx, yy = zip(*[(x,y) for x,y,z in cases if z==flip])
        left.scatter(xx,yy,s=13,alpha=.75,c=color,edgecolors="none",label=tag)
    lim = max(x for x,_,_ in cases)
    left.plot((0,lim),(0,lim),color="#292b33",lw=1,ls="--",label="P/Q tie")
    left.set(xlabel="Original correct-code lead (logits)",ylabel="Shift toward alternative code (logits)",
             title="Phase 2938 | 384 paired views",ylim=(0,max(16,max(y for _,y,_ in cases)*1.1)),xlim=(0,lim*1.02))
    left.legend(loc="upper left", frameon=False)

    labels = {"source":"State vs Note", "clock":"Answer vs early", "content":"Donor vs rotation"}
    group_offsets = {"L21H19": -.12, "trio": .12}
    group_colors = {"L21H19":"#326b95","trio":"#d0584a"}
    ylabels = []
    idx = 0
    for suite in confirm:
        for measure,label in labels.items():
            ylabels.append(f"{suite['suite']} · {label}")
            for g,s in suite["groups"].items():
                test = s["tests"][measure]
                x = test["mean"]
                lo, hi = test["ci95"]
                p = right.errorbar(x,idx+group_offsets[g],xerr=[[x-lo],[hi-x]],fmt="o",
                                   color=group_colors[g],capsize=2,label=g if idx==0 else "_nolegend_")
                if not test["passed"]:
                    p.lines[0].set_markerfacecolor("white")
            idx += 1
    right.set_yticks(range(idx),ylabels)
    right.invert_yaxis()
    right.axvline(0,color="#676e78",lw=1,ls="--")
    right.set(xlabel="Mean P/Q margin contrast (logits)", title="New heldouts | matched reads, open markers fail gate")
    if idx:
        right.legend(loc="lower right",frameon=False)
    for ax in (left,right):
        for side in ("top","right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x",alpha=.18)
    fig.suptitle("Generated-state Value reads in Qwen3-4B",fontweight="bold")
    fig.savefig(R.Base.Previous.check_write(R.OUT/"route-results.png"),dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
