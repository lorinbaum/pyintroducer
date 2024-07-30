import sys
import linecache

class Tracer():
    def __init__(self):
        self.execlines = []
        self.prev_filename = None

    def trace_dispatch(self, frame, event, arg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        line = linecache.getline(filename, lineno)
        if line != "" and \
        "python3.10" not in filename:
            if event == "call":
                self.execlines.append(f"call: {line}")
            if event == "return":
                self.execlines.append(f"retu: {line}")
            if event == "line":
                self.execlines.append(f"line: {line}")
                
        return self.trace_dispatch
        

script_path = sys.argv[1]
tracer = Tracer()
with open(script_path, "r") as script_file:
    script = script_file.read()

sys.settrace(tracer.trace_dispatch)
exec(script)
sys.settrace(None)

output_path = f"{script_path}_trace.txt"
print(f"writing to {output_path}")
with open(output_path, "w") as output_file:
    output_file.write("".join(tracer.execlines))
