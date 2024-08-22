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
# TODO: write OLD in a common comment space and write it also for whitespace
# TODO: move acceptable to lexicon
# TODO: test advance with higher counts. made a mistake with count = 1

class Tracer():
    def __init__(self, output_path):
        self.output_file = open(output_path, "w")
        # stores function and classes. linenumber is where def or class statement is. Also stores lines that have bases (are indented for other reason than class or def)
        self.lexicon:Dict["filename:fileno":str : Dict["parents": List["filename:linenumber":str], "lines": List["lineno":int]]] = {}
        # stores lines by filename:lineno with their "bases": "if", "for", "while", ... any "parent" that caused this line indented that is not class or def.
        # this is for handling visiting different branches inside loop and printing them nicely
        self.bases:Dict["filename":int, Dict["lineno":str, Dict["indent":int, "lineno":int]]] = {}
        # stores all printed lines to avoid duplication
        self.lines:Dict[str:List[int]] = {}
        # stores lines that are part of multilines that should be ignored in all cases
        self.skipmultilines:Dict[str:List[int]] = {}
        # stores filepaths of files that ran through self.preprocess
        self.processedFiles = []
        # class parents that contain nothing except class or function definitions and docstrings
        self.unacceptableParents = []
        self.spaced = True # beginning of line
        self.trueSpaced = False
        self.callStack:List[Dict["name":str, "type": str, "indent": int, "lastIndent": int, "prevlineno": int, "lines": List["lineno":int]]] = []
        # for glob and loca variable printing
        self.namespaces, self.namespace = {}, None
        # lineno and indent for lines != class or def but are followed by an indent
        self.baselines:Dict["filename":str, List["lineno":int]] = {}

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
            if bye["name"] not in self.lexicon: self.lexicon[bye["name"]] = {"parents": [p["name"] for p in currentParents[1:]], "lines": [], "bases":bye["bases"]}

        with open(filename, "r") as f:
            lineno = 0
            baselineno = 1
            lines = f.readline()
            # already holds one entry for "file" so I can store bases in it
            currentParents = [{"name": filename, "indent": -1, "class": False, "acceptable": True, "bases":[]}]
            prevlines, prevlineno, prevIndent = "", 0, 0
            while True:
                lineno += 1
                if lines == "":
                    while currentParents[1:]: farewell()
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
                # process parents, bases
                if (linesS:=lines.strip()) != "":
                    if not linesS.startswith(("class ", "def ", "@", "'''", '"""', "#")) and currentParents[-1]["class"]:
                        currentParents[-1]["acceptable"] = True
                    indent = len(whitespace) if (whitespace:=re.search("^(\s*)", lines)[1]) else 0
                    while indent <= currentParents[-1]["indent"]: farewell()
                    if (cl := linesS.startswith("class ")) or linesS.startswith("def "):
                        classline = linesS.split("\n")[0].split("#")[0].strip() 
                        acceptable = False if not cl or classline.endswith(":") or classline.endswith("pass") else True
                        currentParents.append({"name": f"{filename}:{baselineno}", "indent": indent, "class":cl, "acceptable": acceptable, "bases":[]})
                    while currentParents[-1]["bases"] and indent <= currentParents[-1]["bases"][-1]["indent"]: currentParents[-1]["bases"].pop()
                    if indent > prevIndent and not prevlines.strip().startswith(("class ", "def ")):
                        if filename in self.baselines and self.baselines[filename]: self.baselines[filename].append(prevlineno)
                        else: self.baselines[filename] = [prevlineno]
                        currentParents[-1]["bases"].append({"indent": prevIndent, "lineno": prevlineno})
                    if currentBases := currentParents[-1]["bases"]:
                        if filename not in self.bases: self.bases[filename] = {}
                        self.bases[filename][baselineno] = [b for b in currentBases] # copy the array, otherwise it was overwriting values (?)
                        
                prevlines, prevlineno, prevIndent = lines, baselineno, indent
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
        while nline.strip().startswith(("@", "#")):
            fullParent = [nline, *nmulti] + fullParent
            newlineno = newlineno - 1 - len(nmulti)
            nline, nmulti = self.recede(filename, newlineno, 1)
        # get docstring
        assert (p := f"{filename}:{lineno}") in self.lexicon
        if not self.lexicon[p]["lines"]:
            newlineno = lineno + 1 + len(multilines)
            while True: 
                nline, nmulti = self.advance(filename, newlineno, 0)
                if (nlineS := nline.strip()).startswith(("'''", '"""')):
                    fullParent += [nline, *nmulti]
                    break
                elif not nlineS == "" or nline == "": break
                newlineno = newlineno + 1 + len(nmulti)
        return fullParent

    def introduceParents(self):
        parent = self.callStack[-1]["name"]
        parents = self.lexicon[parent]["parents"] + [parent] # order is old -> young
        self.singleSpace(force = True)
        for p in parents:
            lines = self.getFullParent(*p.split(":"))
            self.write(p.split(":")[0], int(p.split(":")[1]), lines, include_whitespace=False)
            self.spaced = False

    def write(self, filename:str, lineno:int, lines:Union[str, List[str]], glob="{}", loca="{}", old=False, include_whitespace=True):
        # TODO: write should not ask for lines if it has filename and lineno?
        """
        list of lines only if multilines, not intended for list of logically distinct lines
        updates self.callStack[-1]["bases"]
                self.callStack[-1]["lines"]
        """

        # resolve indent
        # assuming "class calls" never happen except in imports, where they are just part of the code and not called from somewhere else, so should not be indented
        if not self.callStack[-1]["lines"] and len(self.callStack) >= 2 and not self.callStack[-1]["type"].startswith("class"): self.callStack[-1]["indent"] += self.tabsize + self.callStack[-2]["lastIndent"]
        self.callStack[-1]["lines"].append(lineno)

        filename_short = filename.split("tinygrad/")[-1] if "tinygrad" in filename else filename
        if len(filename_short) > self.maxfilename: filename_short = f"...{filename_short[-(self.maxfilename-3):]}"
        if not isinstance(lines, list): lines = [lines]
        lineinfo = False
        indent = self.callStack[-1]["indent"]

        # update bases
        lineIndent = len(whitespace) if (whitespace:=re.search("^(\s*)", lines[0])[1]) else 0
        self.callStack[-1]["bases"] = (b if (b:=self.bases.get(filename, {}).get(lineno)) else []) + ([{"lineno": lineno, "indent": lineIndent}] if lineno in self.baselines.get(filename, []) else [])

        # update prevlineno
        self.callStack[-1]["prevlineno"] = lineno

        # get preceding whitespace
        if include_whitespace:
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
                    if line.strip().startswith("yield"): # calls to yield can happen. Find its parent.
                        while not line.strip().startswith("def "):
                            line, multilines = self.recede(filename, lineno, 1)
                            lineno = lineno - 1 - len(multilines)
                    while line.strip().startswith("@"): # a call event will "jump" to the decorator if there is one
                        lineno = lineno + 1 + len(multilines)
                        line, multilines = self.advance(filename, lineno, 0)
                    callType = line.strip()[:5]
                else: callType = ""
                indent, lastIndent = map(lambda x: self.callStack[-1][x] if self.callStack else 0, ["indent", "lastIndent"])
                self.callStack.append({"name": f"{filename}:{lineno}", "type": callType, "indent": indent, "lastIndent": lastIndent, "introduced": [], "lines": [], "bases": []})
            elif event == "return":
                if self.callStack[-1]["lines"]: self.singleSpace()
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

                        # Introduce parents and reprint known function definition if any
                        # can be refactored to only get reintroLines when there an introduction.
                        parent = self.callStack[-1]["name"]
                        if parent in self.lexicon and lineno not in self.lexicon[parent]["lines"] and parent not in self.unacceptableParents:
                            if not self.callStack[-1]["lines"]:
                                self.introduceParents()
                            self.lexicon[parent]["lines"] = sorted(self.lexicon[parent]["lines"] + [lineno])
                            if not line.strip().startswith(("def ", "class ")):
                                if knownLines := self.lexicon[parent]["lines"]:
                                    prevNo = self.callStack[-1]["lines"][-1] if self.callStack[-1]["lines"] else 0
                                    reintroLines = [(no, [ln, *lnmulti]) for no in knownLines if lineno > no > prevNo and no not in self.callStack[-1]["lines"] for ln, lnmulti in [self.advance(filename, no, 0)] if not ln.strip().startswith(("class ", "def ", "@"))]
                                    for no, lines in reintroLines: self.write(filename, no, lines, old=True)
                                # NOTE: if parent is a class, function definition lines when printed will be in its lines list
                                # TODO: should reintroduce only lines from the one preceding onwards. often there is no preceding line, but there might be in a recursive function where the outer function causes a new lines after the children return
                        
                        acceptable = True if not self.callStack or (parent := self.callStack[-1]["name"]) not in self.unacceptableParents else False
                        if not line.strip().startswith(("def ", "class ")) and acceptable:
                            # reintroduce bases if necessary
                            if newBases:=self.bases.get(filename, {}).get(lineno):
                                prevBases, newBases = self.callStack[-1]["bases"], newBases
                                if prevBases != newBases: # bases are different
                                    if lineno > self.callStack[-1]["prevlineno"]:
                                        introduce = [b for b in newBases if b not in prevBases]
                                        for b in introduce:
                                            nline, nmultilines = self.advance(filename, b["lineno"], 0)
                                            self.write(filename, b["lineno"], [nline, *nmultilines])
                                    else:
                                        sharedBase_idx = len([True for b in newBases if b in prevBases]) - 1
                                        assert sharedBase_idx >= 0
                                        nline, nmultilines = self.advance(filename, nlineno:=newBases[sharedBase_idx]["lineno"], 0)
                                        indent = len(whitespace) if (whitespace:=re.search("^(\s*)", nline)[1]) else 0
                                        self.write(filename, nlineno, [f"{indent * ' '}#{nline.strip()} # branches differently:", *nmultilines],  old=True)
                                        for b1, b2 in zip(newBases[sharedBase_idx:], newBases[sharedBase_idx + 1:]):
                                            for no in range(b1["lineno"] + 1, b2["lineno"] + 1):
                                                if (bno:=self.bases.get(filename, {}).get(no)) and bno[-1] == b1:
                                                    nline, nmultilines = self.advance(filename, no, 0)
                                                    self.write(filename, no, [nline, *nmultilines], old=no in self.lines[filename])

                            # print current line
                            self.write(filename, lineno, [line, *multilines], glob=newglob, loca=newloca)


        return self.trace_dispatch

script_path = sys.argv.pop(1)
output_path = f"{script_path}_introduced.py"
tracer = Tracer(output_path)

sys.settrace(tracer.trace_dispatch)
runpy.run_path(script_path, run_name="__main__")
sys.settrace(None)

tracer.output_file.close()
print(f"introduction written to {output_path:30}")