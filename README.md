# Pidentify

Pidentify is an API which aim to classifying with P-value.

This project aims at developing an API to make classifier predict whether a test point belongs to any trained catagory.

If not, classifier will say "none of above".

## Nonlinear square fitting

Using "alglib" library: https://www.alglib.net/interpolation/leastsquares.php#header4

Command for compiling nonlinear.ccp: g++ -I alglib/src -o nonlinear nonlinear.cpp alglib.a
