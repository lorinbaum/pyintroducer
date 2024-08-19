# pyIntroducer

Want to understand code bottom up and expose its ugliness.
Doing this manually with help from a debugger and taking notes becomes impractical as the source code changes a lot.
pyIntroducer.py aims to automate this by generating a readable trace of a script. Outputfile is .py to take advantage of code highlighting.

Differences to default trace are:
- ignores python standard library code
- indentation proportional to callstack depth
- with each call, the callee and its parents, if any, are introduced using the line of their declaration
- Lines (as identified by filename and lineno) are not repeated, unless they are part of
    - callee introductions
    - a function that was previously called that is now being called again but with different lines being executed. In this case, any *known* lines that would execute before the new ones inside that function are reprinted and marked with and `OLD` comment at the end of the line.

    this shortens the output dramatically and enables spanning large differences in abstraction by building on, without repeating, previously defined functionality.
- filename and lineno are printed as comments at the end of the line
- ignores function definitions during imports
- ignores class definitions during imports unless they have any lines that are not part of subclass or function definitions, decorators, docstrings or comments

Currently tried with the tinygrad library, see example outputs in outputs folder.

## More philosophy

The way a library executes, not the way it is structured, is the ground truth for evaluating it - I pay for the compute, organizing it is an extra tool.
The execution can be seen as a story and it better be one that makes sense. Functions and classes are like repeating characters embedded in a dynamic and directing environment.
Readablility of the "introduced" file could be an indicator for simplicity of the traced source code.

Ideally, no extra notes upon reading the trace should be necessary for good understanding. This may require high level comments to provide orientation around the story. I think it should be integrated into the code through docstrings. Which should be perfectly sufficient with the help of a well structured callstack to distinguish high from low level.

## Known problems

- if statements are often misleading because else statements are not printed
- if execution branches differently within for or while loops and a branch is later revisisted, it is unreadable.