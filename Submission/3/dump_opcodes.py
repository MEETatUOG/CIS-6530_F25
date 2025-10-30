# dump_opcodes.py

import os
import csv
import time
from ghidra.program.model.address import AddressSet
from ghidra.app.cmd.disassemble import DisassembleCommand

args = getScriptArgs()

try:
    outputs_folder = os.getcwd()
    results_folder = os.path.join(outputs_folder, 'results')
except Exception as exception:
    print("Exception: {}".format(exception))
    raise

filename = currentProgram.getName()
output_file = os.path.join(results_folder, filename + '.opcode')

try:
    start = time.clock()
    blocks = currentProgram.getMemory().getBlocks()

    if not blocks:
        raise Exception("Exception: no memory block")

    opcodes = []
    for block in blocks:
        section = block.getName()
        address_set = AddressSet(block.getStart(), block.getEnd())

        disassembled = DisassembleCommand(address_set, address_set, True)
        disassembled.applyTo(currentProgram)

        instructions = currentProgram.getListing().getInstructions(address_set, True)
        while instructions.hasNext():
            instruction = instructions.next()
            address = int(instruction.getAddress().getOffset())
            opcode = str(instruction).split(' ')[0]
            opcodes.append([address, opcode, section])

    duration = time.clock() - start

    if not opcodes:
        raise Exception("Exception: no instruction")
    
    os.makedirs(results_folder, exist_ok=True)

    with open(output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['addr', 'opcode', 'section_name'])
        csvwriter.writerows(opcodes)

    print("Opcodes extracted for {}".format(filename))
    print("Duration: {:.2f}s".format(duration))

except Exception as exception:
    print("Exception in extraction: {}".format(exception))
