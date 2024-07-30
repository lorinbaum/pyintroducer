import sys
import linecache
import re
import runpy
from typing import List, Dict, Union
from tokenize import tokenize, TokenError# , IndentationError
from io import BytesIO

# TODO: dictionary definitions print content first, then the variable, which reprints the dictionary because multiline
# TODO: print existing class definition when redefining it
# TODO: its printing whole classes for some reason
# TODO: don't reintroduce just introduced parents
# TODO: don't print empty class declarations
# TODO: program runs absurdly slow. patternmatcher very slow. keeps calling the same functions (?)

# errors = set()

class Tracer():
    def __init__(self, output_path):
        self.output_file = open(output_path, "w")
        
        # stores function and classes. linenumber is where def or class statement is.
        # keep in mind a call event will first go to the decorator if there is one
        # Dict[filename:linenumber : str : Dict[parents: List[filename:linenumber : str], lines: List[line : str]]
        self.lexicon = {}
        # stores all executed lines to avoid duplication
        # new! stores all printed lines, not executed ones
        self.lines:Dict[str:List[int]] = {}
        # stores lines that are part of multilines that should be ignored in all cases
        self.skipmultilines:Dict[str:List[int]] = {}
        # stores current filename:lineno on a call event (def or class statement), so following lines can find their parents easily
        self.parents = []
        # stores parents of the current line when tracing through line events, so that encountered functions and classes can get them assigned
        self.prospectiveParents:List[Dict[str:str, str:int]] = []
        self.parentsIntroduced = False

        self.indent = 0

        self.spaced = True # beginning of line


    def advance(self, filename, lineno, count):
        """
        advances `count` lines from filename:lineno, skipping parts of multiline statements and returns the new line
        advance(filename, lineno, 0) returns full current line with multilines if any
        """
        assert count >= 0
        lines = linecache.getline(filename, lineno)
        if lines.strip() == "": return "", []
        i = 1
        while count + 1:
            try:
                list(tokenize(BytesIO(lines.encode("utf-8")).readline))
            except TokenError as e:
                if str(e).startswith("('EOF in multi-line "):
                    lines += linecache.getline(filename, lineno + i)
                    i += 1
                    continue
            count -= 1
            if count > 0: lines = linecache.getline(filename, lineno + i)
        line, *multilines = [ln + "\n" for ln in lines.split("\n") if ln != ""]
        return line, multilines

    def recede(self, filename, lineno, count):
        """
        recedes `count` lines from filename:lineno, skipping parts of multiline statements and returns the new line
        """
        assert count > 0
        lines = linecache.getline(filename, lineno - 1)
        if lines.strip() == "": return "", []
        i = 2
        magiclines = 0
        while count:
            try:   
                # print("new docstring")
                # print(f"lines when receding through {filename}:{lineno}:{count=}: {lines}")
                list(tokenize(BytesIO(lines.encode("utf-8")).readline))
            except TokenError as e:
                # if e not in errors:
                #     print(e)
                #     errors.add(e)
                if str(e).startswith("('EOF in multi-line "):
                    # if it finds a line with explicit line joining at the end, but the next line is not in lines (it was judges as ok),
                    # it should take that next and see if it tokenizes, error if not
                    if lines.endswith("\\\n") and (nextln := linecache.getline(filename, lineno + magiclines)) not in lines:
                        # print("magic method activate")
                        lines = lines + nextln
                        magiclines += 1
                        continue
                    lines = linecache.getline(filename, lineno - i) + lines
                    i += 1
                    continue
            except IndentationError as e:
                # ignore indentation errors when inside docstrings
                if lines.strip().endswith('"""') or lines.strip().endswith("'''"):
                    lines = linecache.getline(filename, lineno - i) + lines
                    i += 1
                    continue
            count -= 1
            if count > 0: lines = linecache.getline(filename, lineno - i)
        line, *multilines = [ln + "\n" for ln in lines.split("\n") if ln != ""]
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
        if self.parentsIntroduced: return
        parents = [self.parents[-1]] + self.lexicon[self.parents[-1]]["parents"]
        self.singleSpace(force = True)
        for p in parents[::-1]:
            lines = self.getFullParent(*p.split(":"))
            self.write(p.split(":")[0], int(p.split(":")[1]), lines)
        self.spaced = False
        self.parentsIntroduced = True

    def write(self, filename:str, lineno:int, lines:Union[str, List[str]]):
        filename_short = filename.split("tinygrad/")[-1]
        if len(filename_short) > 20: filename_short = f"...{filename_short[-17:]}"
        if not isinstance(lines, list): lines = [lines]
        for ln in lines:
            self.output_file.write(f"{filename_short:20}:{lineno:6}:  {'  ' * self.indent}{ln}")
        self.lines[filename].append(lineno)
        self.trueSpaced = False

    def trace_dispatch(self, frame, event, arg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        if "python3" not in filename:
            if filename not in self.lines: self.lines[filename] = []
            if filename not in self.skipmultilines: self.skipmultilines[filename] = []
            # call events need to go through to let later lines know where they come from
            # same for return events
            # if lineno not in self.lines[filename] or event!="line":
            if lineno not in self.skipmultilines[filename]:
                line, multilines = self.advance(filename, lineno, 0)
                # print([line, *multilines])
                # if event == "line":
                    # for i in range(len(multilines) + 1):
                for i in range(len(multilines)):
                    # self.lines[filename].add(lineno+i)
                    self.skipmultilines[filename].append(lineno+i+1)
                    # print(f"added line {filename}{lineno + i} to visited ones")
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
                                    # line = linecache.getline(filename, lineno + i + len(multilines))
                                    line, multilines = self.advance(filename, lineno, 1)
                                    lineno = lineno + 1 + len(multilines)
                                lineno = lineno - 1 - len(multilines) # to get last line, not next
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
                            if line.strip().startswith("def ") or line.strip().startswith("class "):
                                indent = len(whitespace) if (whitespace:=re.search("^(\s*)", line)[1]) else 0
                                if indent == 0: self.prospectiveParents = [{"name": f"{filename}:{lineno}", "indent": 0}]
                                else:
                                    if indent > (prevIndent:=self.prospectiveParents[-1]["indent"]):
                                        self.prospectiveParents.append({"name": f"{filename}:{lineno}", "indent": indent})
                                    elif indent == prevIndent:
                                        self.prospectiveParents[-1] = {"name": f"{filename}:{lineno}", "indent": indent}
                                    else:
                                        while self.prospectiveParents[-1]["indent"] != indent:
                                            self.prospectiveParents.pop()
                                        self.prospectiveParents[-1] = {"name": f"{filename}:{lineno}", "indent": indent}
                                if (key :=f"{filename}:{lineno}") not in self.lexicon: self.lexicon[key] = {"parents": [p["name"] for p in self.prospectiveParents[:-1]], "lines": []}
                                # if (key :=f"{filename}:{lineno}") not in self.lexicon:
                                #     parents = [] # higher indices = more distant parent
                                #     indent = len(whitespace) if (whitespace:=re.search("^(\s*)", line)[1]) else 0
                                #     newlineno = lineno
                                #     while indent > 0:
                                #         # newline = linecache.getline(filename, lineno - i)
                                #         newline, newmultilines = self.recede(filename, newlineno, 1)
                                #         newlineno = newlineno - 1 - len(newmultilines)
                                #         if newline.strip() != "":
                                #             newIndent = len(whitespace) if (whitespace:=re.search("^(\s*)", newline)[1]) else 0
                                #             if newIndent < indent:
                                #                 parents.append(f"{filename}:{newlineno}")
                                #                 indent = newIndent
                                #     self.lexicon[key] = {
                                #         "parents": parents,
                                #         "lines": []
                                #     }
                            if filename == script_path:
                                self.write(filename, lineno, [line, *multilines])
                                self.spaced = False
                            if not any([
                                line.strip().startswith("@"),
                                line.strip().startswith("from "),
                                line.strip().startswith("import ")
                            ]):
                                if len(self.parents):
                                    if lineno not in self.lexicon[self.parents[-1]]["lines"]:
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
