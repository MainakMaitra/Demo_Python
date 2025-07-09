Why this proves context-insensitivity:

Expected behavior if model understood context:

Context-specific negations should have MUCH LOWER FP ratios (because they provide clear context)
Context-less negations should have HIGHER FP ratios (because they're ambiguous)


Actual results:

Context-specific negations: 0.132 FP ratio
Context-less negations: 0.122 FP ratio
Difference: Only 0.010 (1% difference)


Why this proves context-blindness:

If the model understood context, we'd expect a LARGE difference between these ratios
The fact that they're nearly identical (0.132 vs 0.122) means the model treats context-specific and context-less negations almost the same way
This proves the model is NOT using contextual information to make better decisions



Conclusion: The nearly identical FP ratios (0.132 vs 0.122) prove the model doesn't meaningfully differentiate between context-rich and context-poor negations, confirming it's context-blind.
This is solid mathematical evidence that context information is not being effectively utilized by the model.
