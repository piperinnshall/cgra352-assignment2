if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <C-G>S <Plug>ISurround
imap <C-G>s <Plug>Isurround
imap <C-S> <Plug>Isurround
map! <D-v> *
nnoremap  <Cmd>cclose
nnoremap  :find 
nnoremap  <Cmd>nohlsearch
nnoremap  u <Cmd>UndotreeToggle
nnoremap  ct <Cmd>CTags
nnoremap  e <Cmd>Ex
nnoremap  y "+y
vnoremap  y "+y
xmap S <Plug>VSurround
nnoremap [q :cprevious
nnoremap ]q :cnext
nmap cS <Plug>CSurround
nmap cs <Plug>Csurround
nmap ds <Plug>Dsurround
xmap gS <Plug>VgSurround
nmap gcu <Plug>Commentary<Plug>Commentary
omap gc <Plug>Commentary
nmap gc <Plug>Commentary
xmap gc <Plug>Commentary
xmap gx <Plug>(open-word-under-cursor)
nmap gx <Plug>(open-word-under-cursor)
snoremap gc <Plug>Commentary
nmap gcc <Plug>CommentaryLine
nnoremap g= <Cmd>Format
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
nnoremap z= 1z=
nnoremap <silent> <Plug>SurroundRepeat .
nmap <silent> <Plug>CommentaryUndo :echoerr "Change your <Plug>CommentaryUndo map to <Plug>Commentary<Plug>Commentary"
xnoremap <Plug>(open-word-under-cursor) <ScriptCmd>vim9.Open(getregion(getpos('v'), getpos('.'), { type: mode() })->join())
nnoremap <Plug>(open-word-under-cursor) <ScriptCmd>vim9.Open(GetWordUnderCursor())
nnoremap <C-C> <Cmd>cclose
nnoremap <C-P> :find 
vmap <BS> "-d
vmap <D-x> "*d
vmap <D-c> "*y
vmap <D-v> "-d"*P
nmap <D-v> "*P
imap S <Plug>ISurround
imap s <Plug>Isurround
imap  <Plug>Isurround
iabbr @@ Lorem ipsum dolor sit amet consectetur adipiscing elit quisque faucibus ex sapien vitae pellentesque sem placerat in id cursus mi pretium tellus duis convallis tempus leo eu aenean sed diam urna tempor pulvinar vivamus fringilla lacus nec metus bibendum egestas iaculis massa nisl malesuada lacinia integer nunc posuere ut hendrerit.
let &cpo=s:cpo_save
unlet s:cpo_save
set autowrite
set expandtab
set exrc
set fileencodings=ucs-bom,utf-8,default,latin1
set fillchars=eob:\ ,fold:\ ,foldopen:â”‚,foldsep:â”‚,foldclose:â€º
set helplang=en
set hlsearch
set ignorecase
set incsearch
set laststatus=2
set modelines=0
set path=.,/usr/include,,,**
set runtimepath=~/.vim,~/.vim/pack/plugins/start/vim-surround,~/.vim/pack/plugins/start/vim-commentary,~/.vim/pack/plugins/start/undotree,~/.vim/pack/plugins/start/everforest,/usr/share/vim/vimfiles,/usr/share/vim/vim91,/usr/share/vim/vim91/pack/dist/opt/netrw,/usr/share/vim/vimfiles/after,~/.vim/after
set scrolloff=8
set secure
set shell=/etc/profiles/per-user/piperinnshall/bin/bash
set shiftwidth=2
set noshowmode
set smartcase
set smartindent
set statusline=\ \ \ %f\ %l:%c\ %m
set noswapfile
set tabstop=2
set termguicolors
set ttimeoutlen=0
set undodir=~/.vim/undodir
set undofile
set updatetime=50
set viewoptions=folds
set wildignore=*//target/*
set window=0
" vim: set ft=vim :
