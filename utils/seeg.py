
import re

def split_elec_name(name):
    m = re.match("(^[a-zA-Z]+'?)(\d+)$", name)
    if m is None:
        return None
    else:
        return m.group(1), int(m.group(2))


def make_bipolar(seeg, names):
    seeg_bip = seeg[1:, :] - seeg[:-1, :]
    keep_channels = []
    names_bip = []
    for i in range(len(names) - 1):
        elec1, ind1 = split_elec_name(names[i])
        elec2, ind2 = split_elec_name(names[i+1])
        if elec1 == elec2 and abs(ind1 - ind2) == 1:
            keep_channels.append(i)
            names_bip.append("%s%i-%i" % (elec1, ind2, ind1))
    seeg_bip = seeg_bip[keep_channels, :]
    return seeg_bip, names_bip
