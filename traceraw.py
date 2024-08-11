import sys
import linecache
import runpy

class Tracer():
    def __init__(self):
        self.execlines = []
        self.prev_filename = None
        self.indent = 0

        self.calls, self.retus = 0, 0

    def trace_dispatch(self, frame, event, arg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        line = linecache.getline(filename, lineno)
        if line.strip() != "" and \
        "python3" not in filename and not filename.startswith("<"):
            if event == "call":
                self.execlines.append(f"{self.indent * ' '}call: {line}")
                self.indent += 2
                self.calls += 1
            elif event == "return":
                self.execlines.append(f"{self.indent * ' '}retu: {line}")
                self.indent -= 2
                self.retus += 1
            elif event == "line":
                self.execlines.append(f"{self.indent * ' '}line: {line}")
                
        return self.trace_dispatch
        

# script_path = sys.argv[1]
script_path = "test.py"
tracer = Tracer()

sys.settrace(tracer.trace_dispatch)
runpy.run_path(script_path, run_name="__main__")
sys.settrace(None)

output_path = f"{script_path}_traceraw.txt"
print(f"writing to {output_path}")
with open(output_path, "w") as output_file:
    output_file.write("".join(tracer.execlines))

print(tracer.calls, tracer.retus)