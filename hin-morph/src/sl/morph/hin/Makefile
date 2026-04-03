SHELL = /bin/sh

PREFIX = $(setu)

SUBDIRS = $(PREFIX)/data_src/sl/morph/hin/shell analyser

.PHONY: $(SUBDIRS)

# Program and Data files and directories.
DEST_BIN_PROG_DIR = $(PREFIX)/bin/sl/morph/hin
DEST_BIN_DATA_DIR = $(PREFIX)/data_bin/sl/morph/hin
CPFR = cp -fr

all:

compile: $(SUBDIRS)

$(SUBDIRS):
	export LANG=C
	$(MAKE) -C $@
install:install-exe install-data
install-exe:
	mkdir -p $(DEST_BIN_PROG_DIR)
	$(CPFR) analyser/morph_hin.exe nukta-adder.py $(DEST_BIN_PROG_DIR)
	$(CPFR) adj-gen-1.1 tests docs morph_run.sh morph.sh morph_web.sh morph.spec $(DEST_BIN_PROG_DIR)
install-data:
	mkdir -p $(DEST_BIN_DATA_DIR)/dict
	$(CPFR) $(PREFIX)/data_src/sl/morph/hin/dict/prop_noun_lex.dic $(PREFIX)/data_src/sl/morph/hin/dict/uword.dic $(PREFIX)/data_src/sl/morph/hin/dict/exception_words.txt $(DEST_BIN_DATA_DIR)/dict
clean:
	$(MAKE) -C analyser clean
	rm -fr $(PREFIX)/bin/sl/morph/hin/*
	rm -fr $(PREFIX)/data_bin/sl/morph/hin/*
