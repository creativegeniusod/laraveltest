#!/usr/bin/env python
import sys
import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + "," + str(self.y)


class Vector:
    def __init__(self, pa, pb):
        self.x = int(pb.x) - int(pa.x)
        self.y = int(pb.y) - int(pa.y)

    def __str__(self):
        return str(self.x) + "," + str(self.y)


class Angle:
    def __init__(self, va, vb):
        self.va = va
        self.vb = vb

    def theta(self):
        theta = math.degrees(math.acos((self.va.x * self.vb.x + self.va.y * self.vb.y) / (math.hypot(self.va.x, self.va.y) * math.hypot(self.vb.x, self.vb.y))))
        return theta


class Distance:
    def __init__(self, pa, pb):
        self.x = (int(pb.x) - int(pa.x)) * (int(pb.x) - int(pa.x))
        self.y = (int(pb.y) - int(pa.y)) * (int(pb.y) - int(pa.y))

    def dist(self):
        return (self.x + self.y)**0.5


def checkArg():
    if len(sys.argv) != 2:
        print("please give me file")
        sys.exit(0)


def readFile(filename):
    points = []
    counter = 0
    f = open(filename, "r")
    for line in f.readlines():
        line = line.strip(" \t\n\r")
        if counter == 19:
            break
        x = line.split(",")[0]
        y = line.split(",")[1]
        points.append(Point(x, y))
        counter = counter + 1
    f.close()
    return points


def getCross(va, vb):
    return va.x * vb.y - va.y * vb.x


def getODI(pa, pb, pc, pd, pe, pf, pg, ph):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)

    aa = Angle(va, vb).theta()
    ab = Angle(vc, vd).theta()
    cb = getCross(vc, vd)
    # print cb
    if cb < 0:
        ab = -ab
    # print "u=" + str(aa)
    # print "v=" + str(ab)
    return aa + ab


def getAPDI(pa, pb, pc, pd, pe, pf, pg, ph, pi, pj):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)
    ve = Vector(pi, pj)
    # print vb
    # print vc
    aa = Angle(va, vb).theta()
    ab = Angle(vb, vc).theta()
    ac = Angle(vd, ve).theta()

    cb = getCross(vb, vc)
    cc = getCross(vd, ve)
    # print cb
    if cb > 0:
        ab = -ab
    if cc < 0:
        ac = -ac
    # print "p=" + str(aa)
    # print "q=" + str(ab)
    # print "v=" + str(ac)
    return aa + ab + ac


def writeFile(filename, points, ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype, VBTM_A_type, VBTM_B_type, Yaxistype, MAXILLA_B_type, MANDIBLE_B_type):
    f = open(filename, "w")
    for point in points:
        f.write(str(point) + "\n")
    f.write(ANBtype + "\n")
    f.write(SNBtype + "\n")
    f.write(SNAtype + "\n")
    f.write(ODItype + "\n")
    f.write(APDItype + "\n")
    f.write(FHItype + "\n")
    f.write(FMAtype + "\n")
    f.write(mwtype + "\n")
    f.write(VBTM_A_type + "\n")
    f.write(VBTM_B_type + "\n")
    f.write(Yaxistype + "\n")
    f.write(MAXILLA_B_type + "\n")
    f.write(MANDIBLE_B_type + "\n")
    f.close()


returnString = ""

if __name__ == "__main__":
    checkArg()
    filename = str(sys.argv[1])
    points = readFile(filename)

    # CLASSIFICATION

    # ANB - Angle between A-point (L5), nasion (L2) and B-point (L6).
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[5])
    vc = Vector(points[1], points[0])
    vd = Vector(points[1], points[4])
    ANBtype = ''
    ANB = Angle(vc, vd).theta() - Angle(va, vb).theta()
    if ANB < 3.2:
        ANBtype = '3 - Class III (less than 3.2 degrees)'
    elif ANB > 5.7:
        ANBtype = '2 - Class II (greater than 5.7 degrees)'
    else:
        ANBtype = '1 [Normal] - Class I (between 3.2 to 5.7 degrees)'
    returnString = returnString + "1. ANB Value: " + str(ANB) + " Type:" + ANBtype + " \n"

    # SNB: angle between sella (L1), nasion (L2) and B-point (L6).
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[5])
    SNBtype = ''
    SNB = Angle(va, vb).theta()
    if SNB < 74.6:
        SNBtype = '2 - Retrognatic Mandible (less than 74.6 degrees)'
    elif SNB > 78.7:
        SNBtype = '3 - Prognathic Mandible (greater than 78.7 degrees)'
    else:
        SNBtype = '1 - Normal Mandible (between 74.6 to 78.7 degrees)'
    returnString = returnString + "2. SNB Value: " + str(SNB) + " Type:" + SNBtype + " \n"

    # SNA: angle between sella (L1), nasion (L2) and A-point (L5).
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[4])
    SNAtype = ''
    SNA = Angle(va, vb).theta()
    if SNA < 79.4:
        SNAtype = '3 - Retrognathic Maxilla (less than 79.4 degrees)'
    elif SNA > 83.2:
        SNAtype = '2 - Prognathic Maxilla (greater than 83.2 degrees)'
    else:
        SNAtype = '1 - Normal Maxilla (between 79.4 to 83.2 degrees)'
    returnString = returnString + "3. SNA Value: " + str(SNA) + " Type:" + SNAtype + " \n"

    # Overbite depth indicator (ODI): sum of the angle between the lines from L5 to L6 and from L8 to L10, and the angle between the lines from L3 to L4 and from L17 to L18.
    ODItype = ''
    ODI = getODI(points[7], points[9], points[5], points[4], points[3], points[2], points[16], points[17])
    if ODI < 68.4:
        ODItype = '3 - Open Bite Tendency (less than 68.4 degrees)'
    elif ODI > 80.5:
        ODItype = '2 - Deep Bite Tendency (greater than 80.5 degrees)'
    else:
        ODItype = '1 - Normal (74.5 degrees plus or minus 6.07)'
    returnString = returnString + "4. ODI Value: " + str(ODI) + " Type:" + ODItype + " \n"

    # Anteroposterior dysplasia indicator: sum of the angle between the lines from L3 to L4 and from L2 to L7, the angle between the lines from L2 to L7 and from L5 to L6, and the angle between the lines from L3 to L4 and from L17 to L18.
    APDItype = ''
    APDI = getAPDI(points[2], points[3], points[1], points[6], points[4], points[5], points[3], points[2], points[16], points[17])
    if APDI < 77.6:
        APDItype = '2 - Class II Tendency (less than 77.6 degrees)'
    elif APDI > 85.2:
        APDItype = '3 - Class III Tendency (greater than 85.2 degrees)'
    else:
        APDItype = '1 - Normal (81.4 degrees plus or minus 3.8)'

    returnString = returnString + "5. APDI Value: " + str(APDI) + " Type:" + APDItype + " \n"

    # Facial height index: ratio of the posterior face height (distance from L1 to L10) to the anterior face height (distance from L2 to L8).
    pfh = Distance(points[0], points[9]).dist()
    afh = Distance(points[1], points[7]).dist()
    FHItype = ''
    if pfh / afh < 0.65:
        FHItype = '3 - Long Face Tendency (less than 0.65)'
    elif pfh / afh > 0.75:
        FHItype = '2 - Short Face Tendency (greater than 0.75)'
    else:
        FHItype = '1 - Normal (ratio between 0.65 and 0.75)'
    returnString = returnString + "6. FHI Value: " + str(pfh / afh) + " Type:" + FHItype + " \n"

    # FMA: angle between the lines from sella (L1) to nasion (L2) and from gonion (L10) to gnathion (L9).
    va = Vector(points[0], points[1])
    vb = Vector(points[9], points[8])
    FMAtype = ''
    if Angle(va, vb).theta() < 26.8:
        FMAtype = '3 - Mandible Lower Angle Tendency (less than 26.8 degrees)'
    elif Angle(va, vb).theta() > 31.4:
        FMAtype = '2 - Mandible High Angle Tendency (greater than 31.4 degrees)'
    else:
        FMAtype = '1 - Normal (between 26.8 to 31.4 degrees)'
    returnString = returnString + "7. FMA Value: " + str(Angle(va, vb).theta()) + " Type:" + FMAtype + " \n"

    # Modified Wits Appraisal: ((xL12-xL11)/|xL12-xL11|)||xL12-xL11||.
    # MW = square root((x_{L12} - x_{L11})2 + (y_{L12} - y_{L11})2), if x_{L12} > x_{L11}, a positive MW; otherwise, a negative MW.
    mw = Distance(points[10], points[11]).dist() / 10
    mwtype = ''
    if points[11].x < points[10].x:
        mw = -mw
    if mw >= 2:
        if mw <= 4.5:
            mwtype = '1 - Normal (between 2mm to 4.5mm)'
        else:
            mwtype = '4 - Large Over Jet (MW is greater than 4.5)'
    elif mw == 0:
        mwtype = '2 - Edge to Edge (MW is equal to 0mm)'
    else:
        mwtype = '3 - Anterior Cross Bite (MW is less than 0mm)'
    returnString = returnString + "8. MW Value: " + str(mw) + " Type:" + mwtype + " \n"

    #### Vertical Bite Tendency Measure ####
    # 1 - Palatal Plane (ANS [L17] - PNS [L18]) to Mandibular Plan Angle (Go [L10] - Me [L8])
    va = Vector(points[16], points[17])
    vb = Vector(points[9], points[7])
    VBTM_A_type = ''
    VBTM_A_ = Angle(va, vb).theta()
    if VBTM_A_ < 24:
        VBTM_A_type = '3 - Skeletal closed (less than 24 degrees)'
    elif VBTM_A_ > 33:
        VBTM_A_type = '2 - Skeletal Open (greater than 33 degrees)'
    else:
        VBTM_A_type = '1 - Skeletal Average (between 24 to 33 degrees)'
    returnString = returnString + "9. Palatal to Mandibular (VBT): " + str(VBTM_A_) + " Type:" + VBTM_A_type + " \n"

    # 2 - Angle of Frankfort Horizontal (Por [L4] - O [L3]) to Mandibular Plane (Go [L10] - Me [L8])
    va = Vector(points[3], points[2])
    vb = Vector(points[9], points[7])
    VBTM_B_type = ''
    VBTM_B_ = Angle(va, vb).theta()
    if VBTM_B_ < 18:
        VBTM_B_type = '3 - Skeletal closed (less than 18 degrees)'
    elif VBTM_B_ > 28:
        VBTM_B_type = '2 - Skeletal Open (greater than 28 degrees)'
    else:
        VBTM_B_type = '1 - Skeletal Average (between 18 to 28 degrees)'
    returnString = returnString + "10. Frankfort to Mandibular (VBT): " + str(VBTM_B_) + " Type:" + VBTM_B_type + " \n"

    #### Skeletal Growth Pattern ####
    # 1 - Angle of Frankfort Horizontal (Por [L4] - O [L3]) to (Sella [L1] - Gnathion [L8])
    va = Vector(points[3], points[2])
    vb = Vector(points[0], points[7])
    Yaxis_type = ''
    YAXIS = Angle(va, vb).theta()
    if YAXIS < 57:
        Yaxis_type = '3 - Horizontal Growth (less than 57 degrees)'
    elif YAXIS > 62:
        Yaxis_type = '2 - Vertical Growth (greater than 62 degrees)'
    else:
        Yaxis_type = '1 - Average Growth (between 57 to 62 degrees)'
    returnString = returnString + "11. Y-axis: " + str(YAXIS) + " Type:" + Yaxis_type + " \n"

    # Frankfurt Line equation
    import numpy as np
    px = [int(points[3].x), int(points[2].x)]
    py = [-int(points[3].y), -int(points[2].y)]
    a1, b1 = np.polyfit(px, py, 1)

    # Nasion Perpendicular equation
    a = - (1.0 / a1)
    b = -int(points[1].y) - (a * int(points[1].x))

    #### The Maxilla ####
    # 2 - Distance between the point A [L5] to the Nasion Perpendicular
    #va = Vector(points[3], points[2])
    #vb = Vector(points[1], points[4])

    # X at the Y of L6
    x2 = (-int(points[5].y) - b) / a
    d = int(points[5].x) - x2

    MAXILLA_B_type = ''
    MAXILLA_B_ = d / 10.0
    if MAXILLA_B_ < - 1:
        MAXILLA_B_type = '3 - Retrusive (less than -1 mm)'
    elif MAXILLA_B_ > 3:
        MAXILLA_B_type = '2 - Protrusive (greater than 3 mm)'
    else:
        MAXILLA_B_type = '1 - Average (between -1 to 3 mm)'
    returnString = returnString + "12. Subspinale to Nasion Perpendicular: " + str(MAXILLA_B_) + " Type:" + MAXILLA_B_type + " \n"

    #### The Mandible ####
    # 2 - Distance between the Pogonion [L16] to the Nasion Perpendicular (NP).
    #va = Vector(points[3], points[2])
    #vb = Vector(points[1], points[15])

    # X at the Y of L16
    x2 = (-int(points[15].y) - b) / a
    d = int(points[15].x) - x2

    MANDIBLE_B_type = ''
    MANDIBLE_B_ = d / 10.0
    if MANDIBLE_B_ < - 4:
        MANDIBLE_B_type = '3 - Retrusive (less than -4 mm)'
    elif MANDIBLE_B_ > 1:
        MANDIBLE_B_type = '2 - Protrusive (greater than 1 mm)'
    else:
        MANDIBLE_B_type = '1 - Average (between -4 mm to 1 mm)'
    returnString = returnString + "13. Pogonion to Nasion Perpendicular: " + str(MANDIBLE_B_) + " Type:" + MANDIBLE_B_type + " \n"

    print(returnString)
    #filename = "out-" + filename
    writeFile(filename, points, ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype, VBTM_A_type, VBTM_B_type, Yaxis_type, MAXILLA_B_type, MANDIBLE_B_type)

    def returnstring():
        return returnString
