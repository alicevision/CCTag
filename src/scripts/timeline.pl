#!/usr/bin/perl
# Converts 'perf script' output to data consumable by google charts.

use warnings;
use strict;

#use Data::Dumper;
use JSON;
use Tenjin;

my %times;                    # {tid} -> [samples]
my %funcs;                    # {tid} -> [lvl] -> [[func, b_index, e_eindex]...] (indices of time samples for tid)
my $mainThread;

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
            $startTime ||= $event[2];
            $mainThread ||= $event[1];
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

sub process_thread
{
    my($tid, $data) = @_;
    my $id = scalar @$data;

    foreach my $lvl (0 .. $#{$funcs{$tid}}) {
        foreach my $span (@{$funcs{$tid}->[$lvl]}) {
	    my $item = 
            {
                id => $id++,
                group => int($tid),
		subgroup => $lvl,
                content => $span->[0],
                start => $span->[1],
                end => $span->[2],
		t_b => $times{$tid}[$span->[1]],
		t_e => $times{$tid}[$span->[2]]
            };

	    $item->{'className'} = 'popart' if $span->[0] =~ /popart::/;

	    push @$data, $item;
        }
    }

    return $data;
}

process_input;

my $timeline = [];
my @threads;

foreach my $tid (keys %funcs) {
    process_thread $tid, $timeline;
    push @threads, {
	id => int($tid),
	content => $tid
    };
}

my $json_enc = JSON->new->pretty;
my $tenjin = Tenjin->new();
my $context = {
    main_thread => $mainThread,
    profile_data => $json_enc->encode($timeline),
    threads => $json_enc->encode(\@threads)
};
my $html = $tenjin->render('t1.plhtml', $context);
print $html;
