import sys
import linecache
import re
import runpy
from typing import List, Dict, Union
from tokenize import tokenize, TokenError
from io import BytesIO

# TODO: print existing function definition when taking a different path through it. annotate whats different, note if its complete
    # get complete definitions in preprocessing
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
        # self.parents = []
        # stores parents of the current line when tracing through line events, so that encountered functions and classes can get them assigned
        # self.prospectiveParents:List[Dict[str:str, str:int]] = []
        self.parentsIntroduced:bool = False
        # class parents that contain nothing but class or function definitions and docstrings
        self.unacceptableParents = []
        # dynamic list of currently introduced parents. old -> young
        # self.introduced:List[str] = []

        self.indent = 0
        # stores inherent indent of current line, pyintroducers indent ignored.
        # calls from any such additional indent should consider it when introduced to be sure to differentiate themselves
        self.lastIndent = 0

        self.spaced = True # beginning of line

        # introduced stores introduced parents in the current call
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
        parent = self.callStack[-1]["name"]
        parents = self.lexicon[parent]["parents"][::-1] + [parent] # order is old -> young
        newParents = []
        for i, p in enumerate(parents):
            if p not in self.callStack[-1]["introduced"]:
                newParents = parents[i:]
                break
        if newParents:
            self.singleSpace(force = True)
            # assuming "class calls" never happen except in imports, where they are just part of the code and not called from somewhere else, so should not be indented
            if not self.callStack[-1]["type"].startswith("class") and self.callStack[-2]:
                # if there are any skipped parents, their children will still carry their natural indent, which should be  removed?
                self.callStack[-1]["indent"] += 1 + self.callStack[-2]["lastIndent"]# - skipIndent
            if self.callStack[-1]["introduced"] and self.callStack[-1]["introduced"][0] not in parents:
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
                        self.callStack.append({"name": f"{filename}:{lineno}", "type": line.strip()[:5], "indent": indent, "lastIndent": 0, "introduced": []})
                    elif event == "return":
                        if self.callStack[-1]["type"].startswith("class") or self.callStack[-1]["type"].startswith("def"): self.singleSpace()
                        self.callStack.pop()
                    elif event == "line" and line.strip() != "":
                        """
                        consider the line only if it is not multiline or first line in multiline statement
                        """
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


"""
- dtypes class is not printing even though its acceptable
- not introducing correctly, skips the inner function entirely
    def hook_overflow(dv, fxn):
        def wfxn(*args):
            try: return fxn(*args)
            except OverflowError: return dv
    return wfxn
- callables as arguments are not introduced, like fold_expanded in float4_folding PatternMatcher
- should be able to see how if statements evaluate
"""