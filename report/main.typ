#import "@preview/charged-ieee:0.1.4": ieee

#import "@preview/booktabs:0.0.4": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/lilaq:0.6.0" as lq
#import "@preview/lovelace:0.3.1": *

#show: ieee.with(
  title: [Assignment 2: Advanced Image Editing],
  // abstract: [
  // ],
  authors: (
    (
      name: "Piper Inns Hall",
      department: [NWEN303],
      organization: [Victoria University],
      location: [Wellington],
      email: "innshpipe@myvuw.ac.nz"
    ),
  ),
  // index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  figure-supplement: [Fig.],
)

#show: codly-innit.with()
#codly(
  languages: codly-languages, 
  zebra-fill: none,
  stroke: none,
  display-name: false,
  lang-stroke: none,
  lang-fill: (lang) => white,
)

= 

