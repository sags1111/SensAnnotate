"""
clip_prompts.py
----------------
The CLIP prompt bank used by pipeline.py's classify_with_clip() for
Stage-2 (SAM auto-segmented) regions — objects YOLO's fixed class list
doesn't know about.

Kept in its own file on purpose: extending or rewording this list (the
correct way to fix a missed/misclassified object — see the note below)
should never require touching detection logic in pipeline.py.

One comprehensive prompt bank — no separate special-cased checks (face,
screen, etc). The classification decision in pipeline.py is a pure "which
single prompt matches best" comparison across this entire list; a region
is sensitive if the best-matching prompt happens to be a sensitive one,
non-sensitive otherwise. No manual score cutoff.

Getting classification right is about this list being comprehensive and
well-separated, not about tuning a magic number — if a real-world object
is being missed or misclassified, the fix is almost always to add or
reword a prompt in the right category below, not to add a special-cased
check back into pipeline.py or tweak a threshold.
"""

SENSITIVE_PROMPTS = [
    "a close-up of a human face",
    "a person's body",
    "a person's face looking toward the camera",
    "a partially visible human face",
    "a person's head and face with no covering",
    "a computer monitor displaying code, text, or a terminal window",
    "a laptop screen showing an open application, document, or data",
    "a tablet or handheld device screen showing text or data",
    "a smartphone screen displaying content",
    "a printed document, report, or paper with visible text",
    "a whiteboard or blackboard with handwritten notes, schedules, or diagrams",
    "an open blueprint, technical drawing, or engineering schematic with text",
    "a spreadsheet, chart, or dashboard displayed on a screen",
    "a security camera or CCTV live video feed shown on a monitor",
    "an employee ID card or security badge with a photo and printed text",
    "a keycard, access card, or lanyard with a visible ID badge",
    "an access control keypad or PIN pad display",
    "a biometric access scanner such as a fingerprint or facial recognition device",
    "a SCADA or PLC industrial control system interface on a screen",
    "a factory control room screen displaying operational data",
    "a technical CAD drawing or blueprint displayed on a screen",
    "a vehicle license plate with visible numbers",
    "a machinery serial number plate or nameplate with text",
    "a barcode or QR code label",
    "a shipping label with a name, address, or client information",
    "a package label with a serial number or product code",
    "an invoice, price tag, or purchase order document",
    "an open wallet showing a card or ID",
    "a personal bag or backpack with a visible name tag",
    "a company logo or branding sign that includes readable text",
    "a calendar or schedule poster with names or dates written on it",
    "a person visible from behind machine",
    "a group of people working together",
    "a time clock or attendance punch card with an employee name",
    "a computer screen showing an email inbox or messaging conversation",
    "a signature or handwritten name on a document",
    "a medical record, health form, or patient chart",
    "a salary slip, or HR document",
    "a name tag or name plate on a desk",
    "a passport, ID card, or government-issued document",
    "a phone or video call showing a person's face on a screen",
]

NON_SENSITIVE_PROMPTS = [
    "a person's clothing from behind with no face visible",
    "a person wearing a helmet, gloves, safety vest, or other PPE with face not visible",
    "safety equipment such as a helmet, gloves, or vest not being worn by anyone",
    "an industrial machine body or equipment panel with no visible text",
    "a conveyor belt or assembly line component",
    "an industrial robot arm or automated equipment",
    "generic tools, machine parts, plastic tap, bolts, or screws",
    "a small industrial control button, switch, or indicator light",
    "a small red, green, or yellow LED indicator light",
    "an electrical control panel or cabinet, closed and unlabeled",
    "an air conditioning unit, HVAC compressor, or rooftop cooling unit",
    "electrical cables, wires, or cable bundles",
    "a lathe, milling, drilling, grinding, or CNC machine tool",
    "a welding machine or industrial hydraulic press",
    "hand tools such as a wrench, screwdriver, pliers, or hammer",
    "a mechanical component such as a gear, bearing, bracket, or spring",
    "an industrial valve, pump, or hydraulic hose",
    "a storage tank, drum, dustbin, bin, or industrial container",
    "a forklift, pallet jack, or warehouse cart",
    "a measuring instrument such as a caliper, gauge, or sensor",
    "a plain wall, floor, ceiling Structure or ceiling block",
    "a structural pillar, beam, or metal support frame",
    "an overhead pipe, duct, or ventilation conduit",
    "ceiling lights, lamps, or lighting fixtures",
    "a window, glass panel,mirror or curtain",
    "a painted floor marking or walkway lane",
    "floor, ground, or a reflective surface with no objects",
    "a shadow or light reflection on the floor",
    "an industrial staircase, platform, or handrail",
    "an industrial door, roller shutter, or gate",
    "foam or fabric insulation wrap on an overhead duct or pipe",
    "a table, chair, bottle, bench, or stool",
    "a shelf, rack, dustbin or pallet structure",
    "storage boxes, crates, or wrapped pallets with no visible text",
    "fabric, textile, cloth, or material rolls",
    "a fire extinguisher, fire hydrant, or safety barrier",
    "a generic safety or warning sign with no personal information",
    "a fire alarm or sprinkler system component",
    "a completely black or blank monitor screen with no visible content",
    "a monitor or screen that is turned off, showing only reflection or darkness",
    "a small screw, bolt, or fastener on a machine",
    "a lanyard or neck strap with no visible ID card attached",
]

ALL_PROMPTS = SENSITIVE_PROMPTS + NON_SENSITIVE_PROMPTS