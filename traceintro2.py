import sys
import linecache
import re
import runpy
from typing import List, Dict, Union
from tokenize import tokenize, TokenError
from io import BytesIO

# TODO: print existing function definition when taking a different path through it. annotate whats different, note if its complete
    # get complete definitions in preprocessing
# TODO: don't reintroduce just introduced parents
    # store lastintroduced parent
# TODO: following function calls UOp.const for the first time. the functions definition is skipped.
    # the parent introduction following the call appears to come from nowhere to the unknowing reader
        # def float4_expand_load(load, buf, ex, idx=UOp.const(dtypes.int, 0), idx2=None):
    # maybe print the source of the call if its not printed already and explain what happened.

class Tracer():
    def __init__(self, output_path):
        self.output_file = open(output_path, "w")
        
        # stores function and classes. linenumber is where def or class statement is.
        # keep in mind a call event will first go to the decorator if there is one
        # Dict[filename:linenumber : str : Dict[parents: List[filename:linenumber : str], lines: List[line : str]]
        self.lexicon = {}
        # stores all printed lines to avoid duplication
        self.lines:Dict[str:List[int]] = {}
        # stores lines that are part of multilines that should be ignored in all cases
        self.skipmultilines:Dict[str:List[int]] = {}
        # stores filepaths of file that ran through self.preprocess
        self.processedFiles = []
        # stores current filename:lineno on a call event (def or class statement), so following lines can find their parents easily
        self.parents = []
        # stores parents of the current line when tracing through line events, so that encountered functions and classes can get them assigned
        # self.prospectiveParents:List[Dict[str:str, str:int]] = []
        self.parentsIntroduced = False
        # class parents that contain nothing but class or function definitions and docstrings
        self.unacceptableParents = []

        self.indent = 0

        self.spaced = True # beginning of line


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
        def judgeParent():
            if prospectiveParents and not prospectiveParents[-1]["acceptable"] and prospectiveParents[-1]["class"]:
                self.unacceptableParents.append(prospectiveParents[-1]["name"])
        with open(filename, "r") as f:
            lineno = 0
            baselineno = 1
            lines = f.readline()
            prospectiveParents:List[Dict[str:str, str:int]] = []
            while True:
                lineno += 1
                if lines == "":
                    while prospectiveParents:
                        judgeParent()
                        prospectiveParents.pop()
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
                # from here lines is always a full logical line including multilines if any
                # process parents
                if not any([
                    lines.strip().startswith("class "), lines.strip().startswith("def "), lines.strip().startswith("@"),
                    lines.strip().startswith('"""'), lines.strip().startswith("'''"), lines.strip() == ''
                ]) and len(prospectiveParents) and prospectiveParents[-1]["class"]:
                    # TODO: make it acceptable, if the class is only one line and does not have "pass" after its declaration
                    prospectiveParents[-1]["acceptable"] = True
                if lines.strip().startswith("def ") or lines.strip().startswith("class "):
                    cl = lines.strip().startswith("class ")
                    classline = lines.split("#")[0].strip()
                    acceptable = False if not cl or classline.endswith(":") or classline.endswith("pass") else True
                    indent = len(whitespace) if (whitespace:=re.search("^(\s*)", lines)[1]) else 0
                    assert indent >= 0
                    if not prospectiveParents: prospectiveParents = [{"name": f"{filename}:{baselineno}", "indent": 0, "class":cl, "acceptable": acceptable}]
                    else:
                        if indent > prospectiveParents[-1]["indent"]:
                            prospectiveParents.append({"name": f"{filename}:{baselineno}", "indent": indent, "class":cl, "acceptable": acceptable})
                        elif indent == prospectiveParents[-1]["indent"]:
                            judgeParent()
                            prospectiveParents[-1] = {"name": f"{filename}:{baselineno}", "indent": indent, "class":cl, "acceptable": acceptable}
                        else:
                            while prospectiveParents and indent <= prospectiveParents[-1]["indent"]:
                                judgeParent()
                                prospectiveParents.pop()
                            prospectiveParents.append({"name": f"{filename}:{baselineno}", "indent": indent, "class":cl, "acceptable": acceptable})
                    if (key := f"{filename}:{baselineno}") not in self.lexicon: self.lexicon[key] = {"parents": [p["name"] for p in prospectiveParents[:-1]], "lines": []}
                
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
        if self.parentsIntroduced or self.parents[-1] in self.unacceptableParents: return
        parents = [self.parents[-1]] + self.lexicon[self.parents[-1]]["parents"]
        self.singleSpace(force = True)
        if linecache.getline(self.parents[-1].split(":")[0], int(self.parents[-1].split(":")[1])).strip().startswith("def "): self.output_file.write(f"{' ' * 28}{'  ' * self.indent}# introducing:\n")
        for p in parents[::-1]:
            lines = self.getFullParent(*p.split(":"))
            self.write(p.split(":")[0], int(p.split(":")[1]), lines)
        self.spaced = False
        self.parentsIntroduced = True

    def write(self, filename:str, lineno:int, lines:Union[str, List[str]]):
        filename_short = filename.split("tinygrad/")[-1] if "tinygrad" in filename else filename
        if len(filename_short) > 20: filename_short = f"...{filename_short[-17:]}"
        if not isinstance(lines, list): lines = [lines]
        for ln in lines:
            self.output_file.write(f"{filename_short:20}:{lineno:6}:  {'  ' * self.indent}{ln}")
        self.lines[filename].append(lineno)
        self.trueSpaced = False

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
                        if not any([
                            line.strip().startswith("from "),
                            line.strip().startswith("#")
                        ]):
                            # if call, append to parents "call stack". current parent is latest in there
                            # if return, pop off latest parent
                            # only print def and class lines if they are identical to the current parent
                            if line.strip().startswith("@"):
                                while line.strip().startswith("@"):
                                    line, multilines = self.advance(filename, lineno, 1)
                                    lineno = lineno + 1 + len(multilines)
                            if line.strip().startswith("def ") or line.strip().startswith("class "):
                                self.parentsIntroduced = False
                                self.parents.append(f"{filename}:{lineno}")
                                self.indent += 1
                    elif event == "return":
                        # some calls are not the ones I want, like import lines (or comments!?) which aren't considered for parents
                        # the returns that weren't accounted for happen at end of lines and self.parents will be empty
                        if len(self.parents):
                            self.indent = max(self.indent - 1, 0)
                            self.parents.pop()
                        self.singleSpace()
                    elif event == "line" and line.strip() != "":
                        """
                        consider the line only if it is not multiline or first line in multiline statement
                        """
                        if lineno not in self.lines[filename]:
                            if filename == script_path:
                                self.write(filename, lineno, [line, *multilines])
                                self.spaced = False
                            if not any([
                                line.strip().startswith("@"),
                                line.strip().startswith("from "),
                                line.strip().startswith("import ")
                            ]):
                                if self.parents:
                                    if lineno not in self.lexicon[self.parents[-1]]["lines"] and self.parents[-1] not in self.unacceptableParents:
                                        self.lexicon[self.parents[-1]]["lines"].append(lineno)
                                        if line.strip().startswith("def ") or line.strip().startswith("class "):
                                            if f"{filename}:{lineno}" == self.parents[-1]:
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
