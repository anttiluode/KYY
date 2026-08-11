from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass, asdict


def canonicalize_labels(mapping: tuple[int, ...]) -> tuple[int, ...]:
    table = {}
    nxt = 0
    out = []
    for x in mapping:
        if x not in table:
            table[x] = nxt; nxt += 1
        out.append(table[x])
    return tuple(out)


def dihedral_maps(k: int):
    out = []
    for s in (1,-1):
        for a in range(k):
            out.append(tuple((s*q+a)%k for q in range(k)))
    return sorted(set(out))


def equal_block_quotient(k: int, m: int):
    if k % m: raise ValueError
    r=k//m
    return tuple(q//r for q in range(k))


def compose_state_map(original_to_current: tuple[int,...], current_map: tuple[int,...]):
    return canonicalize_labels(tuple(current_map[x] for x in original_to_current))


def reachable_partitions(n: int):
    """BFS over one-circle instruction set: dihedral permutations + uniform equal-block quotients."""
    start=tuple(range(n)); q=deque([(start,n,[])])
    seen={(start,n)}; records={canonicalize_labels(start):[]}
    while q:
        mapping,k,path=q.popleft()
        for idx,p in enumerate(dihedral_maps(k)):
            nm=compose_state_map(mapping,p); key=(nm,k)
            if key not in seen:
                seen.add(key); q.append((nm,k,path+[f"D{k}:{idx}"]))
                records.setdefault(canonicalize_labels(nm),path+[f"D{k}:{idx}"])
        for m in range(1,k):
            if k%m: continue
            block=equal_block_quotient(k,m)
            nm=compose_state_map(mapping,block); key=(nm,m)
            if key not in seen:
                seen.add(key); q.append((nm,m,path+[f"Q{k}->{m}"]))
                records.setdefault(canonicalize_labels(nm),path+[f"Q{k}->{m}"])
    return records


def harmonic_partition(n: int, harmonic: int):
    """Kernel of phase multiplication phi -> harmonic*phi mod 2pi."""
    h=int(harmonic); g=math.gcd(n,h); order=n//g
    raw=tuple((h*q)%n for q in range(n))
    return canonicalize_labels(raw), {"gcd":g,"output_phase_count":order,"raw_phase_indices":list(raw)}


def contiguous_cyclic_fibers(mapping: tuple[int,...]) -> bool:
    n=len(mapping)
    for lab in set(mapping):
        inds=[i for i,x in enumerate(mapping) if x==lab]
        if len(inds)<=1: continue
        gaps=[]
        s=set(inds)
        transitions=sum(1 for i in range(n) if (i in s) != ((i+1)%n in s))
        if transitions>2: return False
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument("--json",action="store_true");args=p.parse_args()
    rec=reachable_partitions(4)
    alternating=canonicalize_labels((0,1,0,1))
    adjacent=canonicalize_labels((0,0,1,1))
    h2,hmeta=harmonic_partition(4,2)
    payload={
      "one_circle_instruction_set":"dihedral cyclic-order automorphisms + uniform equal-contiguous SHIL quotients",
      "reachable_partition_count_c4":len(rec),
      "adjacent_pair_partition":{"mapping":list(adjacent),"reachable":adjacent in rec,"witness":rec.get(adjacent)},
      "alternating_partition":{"mapping":list(alternating),"reachable":alternating in rec,"contiguous_fibers":contiguous_cyclic_fibers(alternating)},
      "second_harmonic_escape":{"partition":list(h2),"matches_alternating":h2==alternating,**hmeta},
      "theorem":"Every primitive in the one-circle set is cyclic monotone: points may be reordered only by a circle automorphism and collapsed only in contiguous arcs. Composition preserves connected/contiguous fibers, so an interleaved target kernel is impossible in this instruction set.",
      "resource_interpretation":"To realize an interleaved quotient, add a non-order-preserving nonlinear phase instruction such as harmonic multiplication, a nonuniform forcing landscape, or an auxiliary physical state coordinate.",
      "prior_art":"Harmonic/frequency multiplication and multi-phase oscillator logic are established physical primitives; KYY only uses them as backend instructions for a declared transition kernel."
    }
    print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__":main()
