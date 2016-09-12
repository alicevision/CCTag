#!/usr/bin/perl
# Converts 'perf script' output to data consumable by google charts.

use warnings;
use strict;

use Data::Dumper;

my %times;                    # {tid} -> [samples]
my %funcs;                    # {tid} -> [lvl] -> [[func, b_index, e_eindex]...] (indices of time samples for tid)

sub process_input
{
    my $startTime;
    my @event;
    my @bt;

    while (<>) {
        if (!/^\s+/) {          # new stack trace
            @bt = ();
            @event = split /\s+/;
            $event[2] =~ s/://;
            $startTime = $event[2] unless $startTime;
        } elsif (/([0-9a-f]+)\s(.+?)\s\((.+)\)$/) {
            my ($func, $dso) = ($2, $3);
            ($func) = $func =~ /^([^<(]+)/; 
            push @bt, $func if $dso =~ /regression$/;
        }
        elsif (/^$/) {
            process_bt($event[1], $event[2] - $startTime, reverse @bt) if @bt;
        }
    }
}

sub process_bt
{
    my $tid = shift;
    my $sample = shift;

    $times{$tid} ||= [];

    foreach my $lvl (0 .. $#_) {
	my $func = $_[$lvl];
	$funcs{$tid}->[$lvl] ||= [];
	process_fn($tid, $sample, $func, $funcs{$tid}->[$lvl]);
    }

    # Must come last, needed for coalescing of intervals.
    push @{$times{$tid}}, $sample;
}

sub process_fn
{
    my ($tid, $sample, $func, $intervals) = @_;
    my $sample_idx = @{$times{$tid}};

    # Extend interval if no gaps and history exists, otherwise new interval.
    if ($sample_idx && @$intervals && ($intervals->[-1][0] eq $func) && ($intervals->[-1][-1]+1 == $sample_idx)) {
	++$intervals->[-1][-1];
    }
    else {
	push @$intervals, [$func, $sample_idx, $sample_idx];
    }
}

process_input;

print Dumper(%funcs);
 
