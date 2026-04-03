#!/usr/bin/perl
use GDBM_File;
#tie(%LEX,GDBM_File,$ARGV[0],&GDBM_WRCREAT,0644);
tie(%LEX,GDBM_File,$ARGV[0], &GDBM_WRCREAT, 0640) or die "Cannot open:$!\n";
while(<STDIN>) {
$_ =~ /"([^"]+)","(.*)"$/;
if($LEX{$1} eq "") { $LEX{$1} = $2;}
else { $LEX{$1} .= "#".$2;}
}
untie(%LEX);
