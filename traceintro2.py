import sys
import linecache
import re
import runpy
from typing import List, Dict, Union
from tokenize import tokenize, TokenError
from io import BytesIO

# TODO: print existing function definition when taking a different path through it. annotate whats different, note if its complete
    # get complete definitions in preprocessing
    # if a function line is new, also add to lines in current callStack
    # if parent function already has lines in the lexicon, but this line is new:
        # reintroduction?
            # get any lines between current lineno and previous line from callStack[-1]["lines"]
            # print those lines
        # print new line and add NEW to end of line before filename:lineno
        # if this completes the new function, append a line saying so.
# TODO: removet the newParents garbage. since callstack paradigm, there is only one stack of parents per call
# TODO: following function calls UOp.const for the first time. the functions definition is skipped.
    # the parent introduction following the call appears to come from nowhere to the unknowing reader
        # def float4_expand_load(load, buf, ex, idx=UOp.const(dtypes.int, 0), idx2=None):
    # maybe print the source of the call if its not printed already and explain what happened.

class Tracer():
    def __init__(self, output_path):
        self.output_file = open(output_path, "w")
        
        # stores function and classes. linenumber is where def or class statement is.
        # keep in mind a call event will first go to the decorator if there is one
        self.lexicon:Dict["filename:fileno":str : Dict["parents": List["filename:linenumber":str], "lines": List["lineno":int]]] = {}
        # stores all printed lines to avoid duplication
        self.lines:Dict[str:List[int]] = {}
        # stores lines that are part of multilines that should be ignored in all cases
        self.skipmultilines:Dict[str:List[int]] = {}
        # stores filepaths of files that ran through self.preprocess
        self.processedFiles = []
        # class parents that contain nothing except class or function definitions and docstrings
        self.unacceptableParents = []
        self.spaced = True # beginning of line
        # introduced stores introduced parents in the current call. old -> young
        self.callStack:List[Dict["name":str, "type": str, "indent": int, "lastIndent": int, "introduced": List[str]]] = []

        # config

        self.tabsize = 2


    def preprocess(self, filename:str):
        """
        reads file from top to bottom
        multilines:
            tries to tokenize every line. if it fails, its a multiline statement. add lines until tokenize succeeds
            every line in a multiline statement that isn't the first one is added to self.skipmultilines so it can later be ignored
        parents:
            looks at indentation of class and def statements to determine parents, stores current prospective parents
            judges prospective parents before moving to the next.
                blacklists them if they have nothing but class or function definitions and docstrings
        """
        if filename in self.processedFiles: return
        self.processedFiles.append(filename)
        self.skipmultilines[filename] = []
        def farewell():
            """
            pops a parent off currentParents,
            blacklists if if its unacceptable,
            adds it to self.lexicon
            """
            bye = currentParents.pop()
            if not bye["acceptable"] and bye["class"]:
                self.unacceptableParents.append(bye["name"])
            
            if bye["name"] not in self.lexicon: self.lexicon[bye["name"]] = {"parents": [p["name"] for p in currentParents], "lines": [], "fullDef": bye["lines"]}

        with open(filename, "r") as f:
            lineno = 0
            baselineno = 1
            lines = f.readline()
            currentParents:List[Dict["name":str, "indent":int, "class": bool, "acceptable": bool]] = []
            while True:
                lineno += 1
                if lines == "":
                    while currentParents: farewell()
                    break
                # process multilines
                try:
                    list(tokenize(BytesIO(lines.encode("utf-8")).readline))
                except TokenError as e:
                    if "EOF in multi-line" in str(e):
                        self.skipmultilines[filename].append(lineno + 1)
                        lines += f.readline()
                        continue
                    else: raise(e)
                # from here on lines is always a full logical line including multilines if any.
                # process parents
                linesS = lines.strip()
                if linesS != "":
                    if not any([
                        linesS.startswith("class "), linesS.startswith("def "), linesS.startswith("@"),
                        linesS.startswith('"""'), linesS.startswith("'''"), linesS.startswith("#")
                    ]) and currentParents and currentParents[-1]["class"]:
                        currentParents[-1]["acceptable"] = True
                    indent = len(whitespace) if (whitespace:=re.search("^(\s*)", lines)[1]) else 0
                    while currentParents and indent <= currentParents[-1]["indent"]: farewell()
                    if (cl := linesS.startswith("class ")) or linesS.startswith("def "):
                        classline = linesS.split("\n")[0].split("#")[0].strip() 
                        acceptable = False if not cl or classline.endswith(":") or classline.endswith("pass") else True
                        currentParents.append({"name": f"{filename}:{baselineno}", "indent": indent, "class":cl, "acceptable": acceptable, "lines": []})

                    if currentParents and not any([linesS.startswith("class "), linesS.startswith("def "), linesS.startswith("@")]) and not currentParents[-1]["class"]:
                        currentParents[-1]["lines"].append(baselineno)
                
                baselineno = lineno + 1
                lines = f.readline()

    def advance(self, filename, lineno, count):
        """
        advances `count` lines from filename:lineno, skipping parts of multiline statements and returns the new line
        advance(filename, lineno, 0) returns full current line with multilines if any
        """
        assert count >= 0
        assert lineno not in self.skipmultilines[filename]
        newlineno = lineno
        line:str
        multilines = []
        for i in range(count + 1):
            if i == count: line = linecache.getline(filename, newlineno)
            while (newlineno := newlineno + 1) in self.skipmultilines[filename]:
                if i == count: multilines.append(linecache.getline(filename, newlineno))
        return line, multilines

    def recede(self, filename, lineno, count):
        """
        recedes `count` lines from filename:lineno, skipping parts of multiline statements and returns the new line
        """
        assert count > 0
        assert lineno not in self.skipmultilines[filename]
        newlineno = lineno
        line:str
        multilines = []
        for i in range(count):
            while (newlineno := newlineno - 1) in self.skipmultilines[filename]:
                if i == count - 1: multilines.append(linecache.getline(filename, newlineno))
            if i == count - 1: line = linecache.getline(filename, newlineno)
        return line, multilines

    def singleSpace(self, force=False):
        if force and not self.trueSpaced or not self.spaced: self.output_file.write("\n")
        self.spaced = True
        self.trueSpaced = True

    def getFullParent(self, filename:str, lineno:str) -> List[str]:
        lineno = int(lineno)
        line, multilines = self.advance(filename, lineno, 0)
        fullParent = [line, *multilines][::-1]
        line, multilines = self.recede(filename, lineno, 1)
        newlineno = lineno
        while any([line.strip().startswith("@"), line.strip().startswith("#")]):
            fullParent += [line, *multilines]
            newlineno = newlineno - 1 - len(multilines)
            line, multilines = self.recede(filename, newlineno, 1)
        return fullParent[::-1]

    def introduceParents(self):
        parent = self.callStack[-1]["name"]
        parents = self.lexicon[parent]["parents"][::-1] + [parent] # order is old -> young
        newParents = []
        for i, p in enumerate(parents):
            if p not in self.callStack[-1]["introduced"]:
                newParents = parents[i:]
                break
        if newParents:
            if not self.callStack[-1]["everyoneIntroduced"]: self.callStack[-1]["everyoneIntroduced"] = True
            else: print("reintroducing parents in latest callstack") # never triggered
            self.singleSpace(force = True)
            # assuming "class calls" never happen except in imports, where they are just part of the code and not called from somewhere else, so should not be indented
            if not self.callStack[-1]["type"].startswith("class") and self.callStack[-2]:
                self.callStack[-1]["indent"] += 1 + self.callStack[-2]["lastIndent"]
            if self.callStack[-1]["introduced"] and self.callStack[-1]["introduced"][0] not in parents: # not working?
                if linecache.getline(parent.split(":")[0], int(parent.split(":")[1])).strip().startswith("def "):
                    self.output_file.write(f"{'  ' * self.callStack[-1]['indent']}# introducing:\n")
            self.callStack[-1]["introduced"] = parents
            for p in newParents:
                lines = self.getFullParent(*p.split(":"))
                self.write(p.split(":")[0], int(p.split(":")[1]), lines)
            self.spaced = False

    def write(self, filename:str, lineno:int, lines:Union[str, List[str]]):
        filename_short = filename.split("tinygrad/")[-1] if "tinygrad" in filename else filename
        if len(filename_short) > 30: filename_short = f"...{filename_short[-27:]}"
        if not isinstance(lines, list): lines = [lines]
        lineinfo = False
        indent = self.callStack[-1]["indent"]
        for ln in lines:
            if not lineinfo:
                outLine = f"{'  ' * indent}{ln.rstrip():{200 - indent * self.tabsize}} # {filename_short}:{lineno}\n" # fails if indent > 100
                lineinfo = True
            else: outLine = f"{'  ' * indent}{ln}"
            self.output_file.write(outLine)
        self.callStack[-1]["lastIndent"] = int(len(whitespace) / self.tabsize) if (whitespace:=re.search("^(\s*)", lines[0])[1]) else 0
        self.lines[filename].append(lineno)
        self.trueSpaced = False
        self.spaced = False

    def trace_dispatch(self, frame, event, arg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        if "python3" not in filename and not filename.startswith("<"):
            if filename not in self.processedFiles: self.preprocess(filename)
            if filename not in self.lines: self.lines[filename] = []
            if lineno not in self.skipmultilines[filename]:
                line, multilines = self.advance(filename, lineno, 0)
                if line.strip() != "":
                    filename_short = filename.split("tinygrad/")[-1] if "tinygrad" in filename else filename
                    print(f"{filename_short:40}:{lineno:6}", end="\r")
                    if event == "call":
                        if line.strip().startswith("@"):
                            while line.strip().startswith("@"):
                                line, multilines = self.advance(filename, lineno, 1)
                                lineno = lineno + 1 + len(multilines)
                        
                        indent = self.callStack[-1]["indent"] if self.callStack else 0
                        self.callStack.append({"name": f"{filename}:{lineno}", "type": line.strip()[:5], "indent": indent, "lastIndent": 0, "introduced": [], "everyoneIntroduced": False})
                    elif event == "return":
                        if self.callStack[-1]["type"].startswith("class") or self.callStack[-1]["type"].startswith("def"): self.singleSpace()
                        self.callStack.pop()
                    elif event == "line" and line.strip() != "":
                        if lineno not in self.lines[filename]:
                            if not line.strip().startswith("@") or filename == script_path:
                                if self.callStack[-1]["type"].startswith("class") or self.callStack[-1]["type"].startswith("def"):
                                    parent = self.callStack[-1]["name"]
                                    if int(parent.split(":")[1]) not in self.skipmultilines[parent.split(":")[0]]:
                                        if lineno not in self.lexicon[parent]["lines"] and parent not in self.unacceptableParents:
                                            # if self.lexicon[self.parents[-1]]["lines"]: pass
                                                # print(self.lexicon[self.parents[-1]]['lines'])
                                                # its taking a different path in an introduced function
                                                # if the function is currently already introduced
                                                    # max(lines) < lineno
                                                        # add the new line
                                                    # reintroduce the function and note what is happening
                                                # reintroduced the function
                                                # print the previous definition unless
                                            # else: 
                                            # NOTE: if parent is a class, function definition lines when printed will be in its lines list
                                                self.lexicon[parent]["lines"].append(lineno)
                                                if line.strip().startswith("def ") or line.strip().startswith("class "):
                                                    if f"{filename}:{lineno}" == parent:
                                                        self.introduceParents()
                                                else:
                                                    self.introduceParents()
                                                    self.write(filename, lineno, [line, *multilines])
                                                    self.spaced = False
                                elif not line.strip().startswith("def ") and not line.strip().startswith("class "):
                                    self.write(filename, lineno, [line, *multilines])


        return self.trace_dispatch

# script_path = sys.argv[1]
script_path = "test.py"
output_path = f"{script_path}_trace2.txt"
tracer = Tracer(output_path)

sys.settrace(tracer.trace_dispatch)
runpy.run_path(script_path, run_name="__main__")
sys.settrace(None)

tracer.output_file.close()
print(f"introduction written to {output_path:30}")