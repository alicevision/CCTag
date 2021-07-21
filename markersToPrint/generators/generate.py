#!/usr/bin/env python3

import argparse
import os

import svgwrite

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the svg file for the markers.')
    parser.add_argument('--rings', metavar='N', type=int, default=3, choices=[3, 4],
                        help='the number of rings (possible values {%(choices)s}, default: %(default)s)')
    parser.add_argument('--outdir', metavar='dir', type=str, default='./',
                        help='the directory where to save the files (default: %(default)s)')
    parser.add_argument('--margin', metavar='N', type=int, default=400,
                        help='the margin to add around the external ring (default: %(default)s)')
    parser.add_argument('--radius', metavar='N', type=int, default=500,
                        help='the radius of the outer circle (default: %(default)s)')
    parser.add_argument('--addId', action='store_true',
                        help='add the marker id on the top left corner')
    parser.add_argument('--addCross', action='store_true',
                        help='add a small cross in the center of the marker')
    parser.add_argument('--generatePng', action='store_true',
                        help='also generate a png file')
    parser.add_argument('--generatePdf', action='store_true',
                        help='also generate a pdf file')
    parser.add_argument('--whiteBackground', action='store_true',
                        help='set the background (outside the marker) to white instead of transparent')

    args = parser.parse_args()

    # size of the marker (ie the diameter)
    size = 2 * args.radius
    scale = args.radius / 100

    # size of the image, diameter + margin
    width = size + args.margin
    height = size + args.margin

    # id of the first marker
    markerId = 0

    # font size for the id
    font_size = int(0.037 * height)

    # center of the marker
    center = (width / 2, height / 2)

    radius = args.radius

    # create out directory if it does not exist
    if args.outdir and not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    if args.rings == 3:
        input_file = 'cctag3.txt'
    else:
        input_file = 'cctag4.txt'

    with open(input_file) as f:

        for line in f:

            # name of the output file
            base_filename = os.path.join(args.outdir, str(markerId).zfill(4))
            out_filename = base_filename + '.svg'

            # create the svg
            dwg = svgwrite.Drawing(out_filename, profile='tiny', size=(width, height))
            if args.whiteBackground:
                dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='white'))
            # print the id of the marker if required
            if args.addId:
                dwg.add(dwg.text(text=str(markerId), insert=(5, 50), font_size=font_size))

            # print the outer circle as black
            dwg.add(dwg.circle(center=center, r=size / 2, fill='black'))

            fill_color = 'white'
            count = 0
            # each value of the line is the radius of the circle to draw
            # the values are given for a marker of radius 100 (so scale it accordingly to the given size)
            for r in line.split():
                radius = int(r)
                # print(r)
                dwg.add(dwg.circle(center=center, r=scale * radius, fill=fill_color))
                if fill_color == 'white':
                    fill_color = 'black'
                else:
                    fill_color = 'white'
                count = count + 1

            # sanity check
            if args.rings == 3:
                assert count == 5
            else:
                assert count == 7

            if args.addCross:
                # print a small cross in the center
                dwg.add(dwg.line(start=(center[0] - 10, center[1]), end=(center[0] + 10, center[1]), stroke="gray"))
                dwg.add(dwg.line(start=(center[0], center[1] - 10), end=(center[0], center[1] + 10), stroke="gray"))

            dwg.save(pretty=True)

            if args.generatePng or args.generatePdf:
                drawing = svg2rlg(out_filename)
                if args.generatePdf:
                    renderPDF.drawToFile(drawing, base_filename + ".pdf")
                if args.generatePng:
                    renderPM.drawToFile(drawing, base_filename + ".png", fmt="PNG")

            markerId = markerId + 1
