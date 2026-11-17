"""builds data/geant/topology.csv (src,dst,capacity_kbps,weight) from topology-anonymised.xml.
the anonymised file has no capacities and no igp weights, so every link gets weight 1 (hop count) and capacity 1.
the binary link state (congested above the link's own training quantile of utilization) does not depend on a constant capacity."""
import sys
import xml.etree.ElementTree as et


def local(tag):
    return tag.split("}")[-1]


def main(xml_path, out_path):
    root = et.parse(xml_path).getroot()
    edges = set()
    for link in root.find(".//{*}topology").iter():
        if local(link.tag) != "link":
            continue
        a = [e for e in link if local(e.tag) == "from"][0].get("node")
        b = [e for e in link if local(e.tag) == "to"][0].get("node")
        edges.add(tuple(sorted((a, b), key=int)))
    with open(out_path, "w") as f:
        f.write("src,dst,capacity_kbps,weight\n")
        for a, b in sorted(edges, key=lambda e: (int(e[0]), int(e[1]))):
            f.write("%s,%s,1,1\n" % (a, b))
    print(len(edges), "undirected links written to", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
