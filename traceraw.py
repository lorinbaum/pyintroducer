import sys
import linecache
import runpy

class Tracer():
    def __init__(self, opath):
        self.f = open(opath, "w")
        self.execlines = []
        self.prev_filename = None
        self.indent = 0
        
        self.namespaces, self.namespace = {}, None

    def write(self, event, line, filename, lineno, glob, loca):
        globstring = str(glob).replace("\n", "\\n")
        try:
            locastring = str(loca).replace("\n", "\\n")
        except: locastring = f"repr failed"
        self.f.write(f"{self.indent * ' '}{event[:4]:4}: {line.rstrip():{200 - self.indent}}{filename[-30:]}:{lineno:6}    G: {globstring}     L: {locastring}\n")


    def trace_dispatch(self, frame, event, arg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        line = linecache.getline(filename, lineno)
        if line.strip() != "" and \
        "python3" not in filename and not filename.startswith("<"):
            
            if (name := frame.f_globals["__name__"]) != self.namespace: self.namespace = name
            if name not in self.namespaces: self.namespaces[name] = {"glob": {}, "loca": {}}

            newglob = {key:value for key, value in frame.f_globals.items() if key not in ["__builtins__"] and (key not in self.namespaces[name]["glob"] or id(self.namespaces[name]["glob"][key]) != id(value))}
            newloca = {key:value for key, value in frame.f_locals.items() if key not in ["__builtins__", *newglob.keys(), *self.namespaces[name]["glob"].keys()] and (key not in self.namespaces[name]["loca"] or id(self.namespaces[name]["loca"][key]) != id(value))}
            
            self.namespaces[name]["glob"] = frame.f_globals
            self.namespaces[name]["loca"] = frame.f_locals

            self.write(event, line, filename, lineno, newglob, newloca)
            if event == "call": self.indent += 2
            elif event == "return": self.indent -= 2
                
        return self.trace_dispatch
        

script_path = sys.argv.pop(1)
# script_path = "test.py"
output_path = f"{script_path}_traceraw.txt"
tracer = Tracer(output_path)

sys.settrace(tracer.trace_dispatch)
runpy.run_path(script_path, run_name="__main__")
sys.settrace(None)

print(f"writing to {output_path}")
tracer.f.close()