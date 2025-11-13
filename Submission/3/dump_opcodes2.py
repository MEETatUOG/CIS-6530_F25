import os
import csv
import time
from ghidra.program.model.address import AddressSet
from ghidra.app.cmd.disassemble import DisassembleCommand

def main():
    outputs_folder = os.getcwd()
    results_folder = os.path.join(outputs_folder, 'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    filename = currentProgram.getName()
    output_file = os.path.join(results_folder, filename + '.opcode')

    start = time.time()

    listing = currentProgram.getListing()
    memory = currentProgram.getMemory()

    blocks = [b for b in memory.getBlocks() if b.isExecute()]
    if not blocks:
        print("No executable memory blocks found.")
        return

    for block in blocks:
        ins_iter = listing.getInstructions(block, True)
        if not ins_iter.hasNext():
            addrset = AddressSet(block.getStart(), block.getEnd())
            cmd = DisassembleCommand(addrset, None, True)
            cmd.applyTo(currentProgram)

    csvfile = open(output_file, 'wb')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['addr', 'opcode', 'section_name'])

    for block in blocks:
        section_name = block.getName()
        ins_iter = listing.getInstructions(block, True)
        while ins_iter.hasNext():
            inst = ins_iter.next()
            opcode = inst.getMnemonicString()
            address = inst.getAddress().getOffset()
            csvwriter.writerow([address, opcode, section_name])

    csvfile.close()
    duration = time.time() - start

    print("Opcodes extracted for %s" % filename)
    print("Duration: %.2fs" % duration)

if __name__ == "__main__" or True:
    try:
        main()
    except Exception as e:
        print("Exception during opcode extraction: %s" % e)
