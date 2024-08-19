import sys
import linecache
import re
import runpy
from typing import List, Dict, Union
from tokenize import tokenize, TokenError
from io import BytesIO

# TODO: following function calls UOp.const for the first time. the functions definition is skipped.
    # the parent introduction following the call appears to come from nowhere to the unknowing reader
        # def float4_expand_load(load, buf, ex, idx=UOp.const(dtypes.int, 0), idx2=None):
    # maybe print the source of the call if its not printed already and explain what happened.
# TODO: call to hook_overflow not presented correctly?
# TODO: store doc strings in preprocessing and print on first introduction
# TODO: store comments and empty lines in preprocessing, if comment, store with indent of the comment, if empty line, store with indent of the previous code line
    # when writing, check if there are any comments / empty lines with the indent of the new line between previously printed an new line

class Tracer():
    def __init__(self, output_path):
        self.output_file = open(output_path, "w")
        
        # stores function and classes. linenumber is where def or class statement is.
        self.lexicon:Dict["filename:fileno":str : Dict["parents": List["filename:linenumber":str], "lines": List["lineno":int], "whitespace": List[Dict["indent":int, "line":str], "doc": str]]] = {}
        # stores all printed lines to avoid duplication
        self.lines:Dict[str:List[int]] = {}
        # stores lines that are part of multilines that should be ignored in all cases
        self.skipmultilines:Dict[str:List[int]] = {}
        # stores filepaths of files that ran through self.preprocess
        self.processedFiles = []
        # class parents that contain nothing except class or function definitions and docstrings
        self.unacceptableParents = []
        self.spaced = True # beginning of line
        self.callStack:List[Dict["name":str, "type": str, "indent": int, "lastIndent": int], "lines": List["lineno":int]] = []
        # for glob and loca variable printing
        self.namespaces, self.namespace = {}, None

        # config

        self.tabsize = 2
        self.maxfilename = 30


    def preprocess(self, filename:str):
        """
        reads file from top to bottom
        multilines:
            tries to tokenize every line. if it fails, its a multiline statement. add lines until tokenize succeeds
            every line in a multiline statement that isn't the first one is added to self.skipmultilines so it can later be ignored
        parents:
            looks at indentation of class and def statements to determine parents, stores current parents before moving to the next.
            blacklists them if they have nothing but class or function definitions and docstrings
        """
        if filename in self.processedFiles: return
        self.processedFiles.append(filename)
        self.skipmultilines[filename] = []
        def farewell():
            """
            pops a parent off currentParents, blacklists it if its unacceptable, adds it to self.lexicon
            """
            if not (bye := currentParents.pop())["acceptable"] and bye["class"]: self.unacceptableParents.append(bye["name"])
            if bye["name"] not in self.lexicon: self.lexicon[bye["name"]] = {"parents": [p["name"] for p in currentParents], "lines": []}

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
                        currentParents.append({"name": f"{filename}:{baselineno}", "indent": indent, "class":cl, "acceptable": acceptable})
                
                baselineno = lineno + 1
                lines = f.readline()

    def advance(self, filename, lineno, count):
        """
        advances `count` lines from filename:lineno, skipping parts of multiline statements and returns the new line
        advance(filename, lineno, 0) returns full current line with multilines if any
        """
        assert count >= 0
        assert lineno not in self.skipmultilines[filename], f"{filename=}, {lineno=}"
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
        """
        returns lines including comments and decorators immediately before the function definition, function definition itself and
        its docstring if not self.lexicon[parent]["lines"]
        """
        lineno = int(lineno)
        line, multilines = self.advance(filename, lineno, 0)
        fullParent = [line, *multilines]
        nline, nmulti = self.recede(filename, lineno, 1)
        newlineno = lineno
        while any([nline.strip().startswith("@"), nline.strip().startswith("#")]):
            fullParent = [nline, *nmulti] + fullParent
            newlineno = newlineno - 1 - len(nmulti)
            nline, nmulti = self.recede(filename, newlineno, 1)
        # get docstring
        assert (p := f"{filename}:{lineno}") in self.lexicon
        if not self.lexicon[p]["lines"]:
            newlineno = lineno + 1 + len(multilines)
            while True: 
                nline, nmulti = self.advance(filename, newlineno, 0)
                if (nlineS := nline.strip()).startswith("'''") or nlineS.startswith('"""'):
                    fullParent += [nline, *nmulti]
                    break
                elif not nlineS == "" or nline == "": break
                newlineno = newlineno + 1 + len(nmulti)


        return fullParent

    def introduceParents(self):
        parent = self.callStack[-1]["name"]
        parents = self.lexicon[parent]["parents"] + [parent] # order is old -> young
        self.singleSpace(force = True)
        # assuming "class calls" never happen except in imports, where they are just part of the code and not called from somewhere else, so should not be indented
        if not self.callStack[-1]["type"].startswith("class") and len(self.callStack) >= 2: self.callStack[-1]["indent"] += self.tabsize + self.callStack[-2]["lastIndent"]
        for p in parents:
            lines = self.getFullParent(*p.split(":"))
            self.write(p.split(":")[0], int(p.split(":")[1]), lines, whitespace=False)
            self.spaced = False

    def write(self, filename:str, lineno:int, lines:Union[str, List[str]], glob="{}", loca="{}", old=False, whitespace=True):
        filename_short = filename.split("tinygrad/")[-1] if "tinygrad" in filename else filename
        if len(filename_short) > self.maxfilename: filename_short = f"...{filename_short[-(self.maxfilename-3):]}"
        if not isinstance(lines, list): lines = [lines]
        lineinfo = False
        indent = self.callStack[-1]["indent"]

        # get preceding whitespace
        if whitespace:
            nlineno, newlines = lineno, []
            while True:
                nline, nmulti = self.recede(filename, nlineno, 1)
                if (nlineS := nline.strip()).startswith("#") or nlineS == "" and nlineno >= 1:
                    newlines = [nline, *nmulti] + newlines
                    nlineno = nlineno - 1 - len(nmulti)
                else: break
            for ln in newlines: self.output_file.write(f"{' ' * indent}{ln.rstrip()}\n")

        globstring = str(glob).replace("\n", "\\n")
        try:
            locastring = str(loca).replace("\n", "\\n")
        except: locastring = f"repr failed"

        for ln in lines:
            if not lineinfo:
                if old: 
                    outLine = f"{' ' * indent}{ln.rstrip():{200 - indent - 6}} # OLD # {filename_short:{self.maxfilename}}:{lineno:6}\n" # fails if indent > 200
                else:
                    outLine = f"{' ' * indent}{ln.rstrip():{200 - indent}} # {filename_short:{self.maxfilename}}:{lineno:6}    G: {globstring}     L: {locastring}\n" # fails if indent > 200
                lineinfo = True
            else:
                outLine = f"{' ' * indent}{ln}"
            self.output_file.write(outLine)
        self.callStack[-1]["lastIndent"] = len(whitespace) if (whitespace:=re.search("^(\s*)", lines[0])[1]) else 0
        self.lines[filename].append(lineno)
        self.trueSpaced = False
        self.spaced = False

    def trace_dispatch(self, frame, event, arg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        if "python3" not in filename and not filename.startswith("<"):
            if filename not in self.processedFiles: self.preprocess(filename)
            if filename not in self.lines: self.lines[filename] = []
            if lineno not in self.skipmultilines[filename]: line, multilines = self.advance(filename, lineno, 0)
            else: line = None
            if event == "call":
                if line != None:
                    # a call event will "jump" to the decorator if there is one
                    if line.strip().startswith("@"):
                        while line.strip().startswith("@"):
                            line, multilines = self.advance(filename, lineno, 1)
                            lineno = lineno + 1 + len(multilines)
                    callType = line.strip()[:5]
                else: callType = ""
                indent, lastIndent = map(lambda x: self.callStack[-1][x] if self.callStack else 0, ["indent", "lastIndent"])
                self.callStack.append({"name": f"{filename}:{lineno}", "type": callType, "indent": indent, "lastIndent": lastIndent, "introduced": [], "lines": []})
            elif event == "return":
                if self.callStack[-1]["type"].startswith("class") or self.callStack[-1]["type"].startswith("def"): self.singleSpace()
                self.callStack.pop()
            elif event == "line" and lineno not in self.skipmultilines[filename] and line.strip() != "":
                if lineno not in self.lines[filename]:
                    if not line.strip().startswith("@") or filename == script_path:
                        # global, local variables
                        if (name := frame.f_globals["__name__"]) != self.namespace: self.namespace = name
                        if name not in self.namespaces: self.namespaces[name] = {"glob": {}, "loca": {}}
                        newglob = {key:value for key, value in frame.f_globals.items() if key not in ["__builtins__"] and (key not in self.namespaces[name]["glob"] or id(self.namespaces[name]["glob"][key]) != id(value))}
                        newloca = {key:value for key, value in frame.f_locals.items() if key not in ["__builtins__", *newglob.keys(), *self.namespaces[name]["glob"].keys()] and (key not in self.namespaces[name]["loca"] or id(self.namespaces[name]["loca"][key]) != id(value))}
                        self.namespaces[name]["glob"] = frame.f_globals
                        self.namespaces[name]["loca"] = frame.f_locals

                        if self.callStack[-1]["type"].startswith("class") or self.callStack[-1]["type"].startswith("def"):
                            parent = self.callStack[-1]["name"]
                            # assert parent in self.lexicon, f"{parent}"
                            if parent in self.lexicon and lineno not in self.lexicon[parent]["lines"] and parent not in self.unacceptableParents:
                                if not self.callStack[-1]["lines"]: self.introduceParents()
                                self.callStack[-1]["lines"].append(lineno)
                                self.lexicon[parent]["lines"].append(lineno)
                                if not line.strip().startswith("def ") and not line.strip().startswith("class "):
                                    if knownLines := self.lexicon[parent]["lines"]:
                                        prevNo = self.callStack[-1]["lines"][-2] if self.callStack[-1]["lines"] and len(self.callStack[-1]["lines"]) > 1 else 0
                                        # print all lines from previous line in callstack to current line
                                        for no in knownLines:
                                            if lineno > no > prevNo:
                                                assert no not in self.skipmultilines[filename], f"{filename=}, {lineno}, {parent=}, {knownLines=}"
                                                knownLine, knownmultilines = self.advance(filename, no, 0)
                                                self.write(filename, no, [knownLine, *knownmultilines], old=True)
                                            elif no >= lineno: break
                                    # NOTE: if parent is a class, function definition lines when printed will be in its lines list
                                    if not line.strip().startswith("def ") and not line.strip().startswith("class "):
                                        self.write(filename, lineno, [line, *multilines], glob=newglob, loca=newloca)
                        elif not line.strip().startswith("def ") and not line.strip().startswith("class "):
                            self.write(filename, lineno, [line, *multilines], glob=newglob, loca=newloca)


        return self.trace_dispatch

script_path = sys.argv.pop(1)
output_path = f"{script_path}_introduced.py"
tracer = Tracer(output_path)

sys.settrace(tracer.trace_dispatch)
runpy.run_path(script_path, run_name="__main__")
sys.settrace(None)

tracer.output_file.close()
print(f"\033[1G\033[2Kintroduction written to {output_path:30}")