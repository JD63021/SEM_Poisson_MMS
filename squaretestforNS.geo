// Gmsh project created on Mon Sep 08 09:10:28 2025
SetFactory("OpenCASCADE");
//+
SetFactory("Built-in");
//+
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Physical Curve(5) = {4, 3, 2, 1};
//+
Physical Surface(6) = {1};
//+
Transfinite Curve {4, 3, 2, 1} = 3 Using Progression 1;
//+
Transfinite Surface {1};

