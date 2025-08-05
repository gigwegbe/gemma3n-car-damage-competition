ans = """🔧 Ranked Detections by Severity:
glass shatter | Conf: 0.62 | Area: 339997 | Score: 944693 | BBox: (179.0260009765625, 136.43240356445312, 819.842529296875, 667.0)
scratch | Conf: 0.35 | Area: 82880 | Score: 43127 | BBox: (585.1806030273438, 82.36738586425781, 999.3623046875, 282.4722900390625)"""

instruction_template = """You are an expert automobile inspector.
Review the image and verify whether the car damages shown are consistent
with the following object detection results (ranked by severity):

ans{}

Mention the damage types and whether they match what you see in the image.
"""

instruction = instruction_template.format(ans)

print(instruction)