#!/usr/bin/perl
# Converts 'perf script' output to data consumable by google charts.

use warnings;
use strict;

use Data::Dumper;

# Call stacks can be incomplete, so we disregard stack level.
# TODO: use sample index start/end; easier than times!
my %times;                    # {tid} -> [samples]
my %funcs;                    # {tid} -> {func} -> [[begin, end, first_sample_index, count], ...]

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
            $func =~ s/\(.+//;
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

    foreach my $func (@_) {
        $funcs{$tid}->{$func} ||= [];
        process_fn($tid, $sample, $funcs{$tid}->{$func});
    }

    push @{$times{$tid}}, $sample;
}

sub process_fn
{
    my ($tid, $sample, $intervals) = @_;

    if (!@{$times{$tid}} || !@{$intervals}) {
        push @$intervals, [$sample, $sample, scalar @{$times{$tid}}, 1];
        return;
    }

    # No gap between all samples and samples for this fn; extend interval
    if ($intervals->[-1][1] == $times{$tid}->[-1]) {
        $intervals->[-1][1] = $sample;
        ++$intervals->[-1][3];
        return;
    }

    # Start new interval.
    push @$intervals, [$sample, $sample, scalar @{$times{$tid}}, 1];
}

process_input;

print Dumper(%funcs);
 
